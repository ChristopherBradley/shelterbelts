"""
Upload a folder of GeoTIFFs to a GEE ImageCollection, staging through a Google Cloud Storage bucket. 
Auto-deletes each Cloud storage object after uploaded to GEE to avoid storage costs.

Example (2025 ag predictions):
  python upload_to_gee.py \
      /scratch/xe2/cb8590/barra_trees_s4_ag_noxy_df_4326_2025/subfolders \
      --suffix _merged_predicted.tif \
      --bucket cb8590-shelterbelts-gee \
      --gcs-prefix barra_trees_s4_ag_noxy_df_4326_2025 \
      --collection projects/ee-christopher-bradley/assets/Aus2025_ag_noxy_predictions \
      --project ee-christopher-bradley \
      --key ~/gee-uploader-key.json

The same authentication is used for both Earth Engine and Cloud Storage with a single service account.

"""

import os
import glob
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
from google.cloud import storage


def _service_account_email(key_file):
    with open(key_file) as f:
        return json.load(f)['client_email']


def init_clients(project, key_file=None):
    """Authenticate Earth Engine + Cloud Storage from the same credential."""
    if key_file is None:
        key_file = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

    if key_file:
        email = _service_account_email(key_file)
        ee.Initialize(ee.ServiceAccountCredentials(email, key_file), project=project)
        gcs = storage.Client.from_service_account_json(key_file, project=project)
        print(f"Authenticated as service account {email}")
    else:
        # Ambient credentials (earthengine authenticate + gcloud application-default login)
        ee.Initialize(project=project)
        gcs = storage.Client(project=project)
        print("Authenticated with ambient/default credentials")
    return gcs


def ensure_collection(collection_id):
    """Create the ImageCollection asset if it doesn't already exist."""
    try:
        ee.data.getAsset(collection_id)
        print(f"Collection already exists: {collection_id}")
    except ee.EEException:
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, collection_id)
        print(f"Created collection: {collection_id}")


def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except ee.EEException:
        return False


def upload_one_to_gcs(bucket, blob_name, local_path, retries=4):
    """Upload a single file to GCS, retrying transient failures; never raises.

    Returns (blob_name, status) where status is 'uploaded', 'skipped (already in bucket)',
    or 'FAILED: <reason>'. A transient network/timeout on one file must not abort the whole
    run (that is exactly what used to kill a batch near completion), so we retry with
    exponential backoff and, on final failure, report the error rather than raising it.
    """
    blob = bucket.blob(blob_name)
    try:
        if blob.exists():
            blob.reload()
            if blob.size == os.path.getsize(local_path):
                return blob_name, 'skipped (already in bucket)'
    except Exception:
        pass  # existence pre-check is a best-effort fast-path; fall through and (re)upload

    last_err = None
    for attempt in range(retries):
        try:
            blob.upload_from_filename(local_path)
            return blob_name, 'uploaded'
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # backoff: 1s, 2s, 4s ...
    return blob_name, f'FAILED: {type(last_err).__name__}: {str(last_err)[:150]}'


def _delete_blob(bucket, blob_name):
    try:
        bucket.blob(blob_name).delete()
    except Exception:
        pass  # already gone / never uploaded


def _collection_asset_ids(collection_id):
    """Every asset id currently in the collection (authoritative Cloud API, paginated)."""
    ids, params = set(), {'parent': collection_id}
    while True:
        resp = ee.data.listAssets(params)
        ids.update(a['id'] for a in resp.get('assets', []))
        token = resp.get('nextPageToken')
        if not token:
            return ids
        params['pageToken'] = token


def wait_and_cleanup(collection_id, pending, bucket, poll_interval=30, stall_timeout=3600):
    """Delete each staging blob the moment its image actually lands in the collection.

    pending: dict of image_id (full asset id) -> blob_name.

    We poll the collection's asset list (the authoritative Cloud API) rather than
    ee.data.getTaskStatus: that legacy status endpoint lagged badly during a real run —
    it kept reporting tasks as still-running for the whole wait window even though the
    Cloud API showed them SUCCEEDED and the assets had landed. Presence in the collection
    is the ground truth.

    Rather than a flat wall-clock deadline (which falsely reported "0 ingested" whenever
    EE's ingestion queue was merely slow), we wait as long as images keep landing and only
    give up once no new image has appeared for `stall_timeout` seconds. Anything still
    missing when progress stalls keeps its staging blob — the ingestion task is usually
    still queued server-side and lands later; re-running is idempotent and finishes the
    cleanup, and the bucket lifecycle rule is the final backstop.
    """
    pending = dict(pending)
    total = len(pending)
    n_done = 0
    last_progress = time.time()
    print(f"\nWaiting for {total} images to finish ingesting, deleting staging files as they land...")
    print(f"  (keeps waiting while images keep landing; gives up only after {stall_timeout}s with no progress)")
    while pending:
        present = _collection_asset_ids(collection_id)
        landed = [image_id for image_id in pending if image_id in present]
        for image_id in landed:
            _delete_blob(bucket, pending.pop(image_id))
            n_done += 1
            if n_done % 20 == 0:
                print(f"  {n_done}/{total} ingested + staging deleted...", flush=True)
        if landed:
            last_progress = time.time()
        elif time.time() - last_progress > stall_timeout:
            break
        if pending:
            time.sleep(poll_interval)

    if pending:
        print(f"\nStopped waiting: {n_done}/{total} ingested, {len(pending)} not yet present after "
              f"{stall_timeout}s with no further progress.")
        print("  This is NOT a failure — the ingestion tasks were started successfully and are likely "
              "still queued server-side (they often land minutes-to-hours later). Their staging files "
              "are kept; re-run this same command to finish cleanup (idempotent), or check "
              "`earthengine task list`.")
    else:
        print(f"\nIngestion complete: all {n_done}/{total} images ingested + staging deleted.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', help='Folder containing the tifs to upload')
    parser.add_argument('--suffix', default='.tif', help="Only upload tifs ending with this (default: .tif)")
    parser.add_argument('--bucket', required=True, help='GCS bucket name (without gs://)')
    parser.add_argument('--gcs-prefix', default='gee_upload', help='Object-name prefix (folder) inside the bucket')
    parser.add_argument('--collection', required=True, help='Full EE ImageCollection asset id')
    parser.add_argument('--project', required=True, help='Google Cloud / EE project id')
    parser.add_argument('--key', default=None, help='Service-account json key')
    parser.add_argument('--pyramiding', default='MODE', help='Pyramiding policy: MEAN/MODE/SAMPLE/MIN/MAX (default: MODE)')
    parser.add_argument('--nodata', type=float, default=None, help='Optional no-data value to mask on ingestion')
    parser.add_argument('--workers', type=int, default=8, help='Parallel GCS upload workers (default: 8)')
    parser.add_argument('--keep-staging', action='store_true', help="Don't delete staging files after ingestion (default: delete on completion)")
    parser.add_argument('--poll-interval', type=int, default=30, help='Seconds between task-status polls (default: 30)')
    parser.add_argument('--stall-timeout', '--timeout', dest='stall_timeout', type=int, default=3600,
                        help='Give up waiting only after this many seconds with NO new image landing '
                             '(default: 3600). As long as images keep appearing it keeps waiting, so a '
                             'slow EE ingestion queue no longer causes a premature "0 ingested" report.')
    parser.add_argument('--dry-run', action='store_true', help='List what would happen without uploading/ingesting')
    args = parser.parse_args()

    tifs = sorted(f for f in glob.glob(os.path.join(args.folder, '*.tif')) if f.endswith(args.suffix))
    print(f"Found {len(tifs)} tifs ending with '{args.suffix}' in {args.folder}")
    if not tifs:
        return

    if args.dry_run:
        print(f"[dry-run] Would upload to gs://{args.bucket}/{args.gcs_prefix}/ and ingest into {args.collection}")
        for f in tifs[:5]:
            print(f"  e.g. {Path(f).name} -> {args.collection}/{Path(f).stem}")
        if len(tifs) > 5:
            print(f"  ... and {len(tifs) - 5} more")
        return

    gcs = init_clients(args.project, args.key)
    bucket = gcs.bucket(args.bucket)
    ensure_collection(args.collection)

    # --- Stage 1: upload tifs to GCS in parallel ---
    print(f"\nUploading {len(tifs)} tifs to gs://{args.bucket}/{args.gcs_prefix}/ ...")
    gcs_uris, blob_names, failed_uploads = {}, {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for f in tifs:
            blob_name = f"{args.gcs_prefix}/{Path(f).name}"
            blob_names[f] = blob_name
            gcs_uris[f] = f"gs://{args.bucket}/{blob_name}"
            futures[pool.submit(upload_one_to_gcs, bucket, blob_name, f)] = f
        for i, fut in enumerate(as_completed(futures), 1):
            f = futures[fut]
            blob_name, status = fut.result()  # upload_one_to_gcs never raises
            if status.startswith('FAILED'):
                failed_uploads.append(f)
            if i % 20 == 0 or status == 'uploaded' or status.startswith('FAILED'):
                print(f"  [{i}/{len(tifs)}] {blob_name}: {status}", flush=True)

    if failed_uploads:
        print(f"\n{len(failed_uploads)} file(s) failed to upload after retries and will be skipped for "
              f"ingestion this run (re-run the same command to retry them — it's idempotent):")
        for f in failed_uploads[:10]:
            print(f"  - {Path(f).name}")
        if len(failed_uploads) > 10:
            print(f"  ... and {len(failed_uploads) - 10} more")
        # Only ingest the files that actually made it into the bucket.
        tifs = [f for f in tifs if f not in set(failed_uploads)]
        if not tifs:
            print("No files were successfully staged; nothing to ingest.")
            return

    # --- Stage 2: kick off EE ingestion for each tif ---
    print(f"\nIngesting into {args.collection} ...")
    pending, n_skipped = {}, 0  # pending: image_id -> blob_name (staging to delete once ingested)
    for f in tifs:
        image_id = f"{args.collection}/{Path(f).stem}"
        if asset_exists(image_id):
            n_skipped += 1
            if not args.keep_staging:
                _delete_blob(bucket, blob_names[f])  # already ingested; staging not needed
            continue
        manifest = {
            'name': image_id,
            'tilesets': [{'sources': [{'uris': [gcs_uris[f]]}]}],
            'pyramidingPolicy': args.pyramiding,
        }
        if args.nodata is not None:
            manifest['missingData'] = {'values': [args.nodata]}
        task_id = ee.data.newTaskId()[0]
        ee.data.startIngestion(task_id, manifest, allow_overwrite=True)
        pending[image_id] = blob_names[f]
        time.sleep(0.1)  # gentle on the API

    print(f"Ingestion tasks started: {len(pending)}, already-present images skipped: {n_skipped}")

    # --- Stage 3: wait + delete staging as each image lands in the collection ---
    if args.keep_staging:
        print("Leaving staging files in the bucket (--keep-staging). Track tasks with: earthengine task list")
        return
    wait_and_cleanup(args.collection, pending, bucket, args.poll_interval, args.stall_timeout)


if __name__ == '__main__':
    main()
