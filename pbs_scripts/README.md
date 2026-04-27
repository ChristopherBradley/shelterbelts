# pbs_scripts

PBS job scripts and bash launchers that run the shelterbelts pipeline at
continent scale on NCI Gadi. Each stage of the pipeline has a `.pbs` file
(the job script submitted to the scheduler) and a matching `.sh` launcher
(fans out work across tiles / years by calling `qsub -v ...` per unit).

The numbered workflow below is the canonical order. Every stage writes its
output to `/scratch/xe2/cb8590/...` so the next stage can pick it up.

## Workflow

| # | Stage | Script(s) | Module it runs | Inputs | Outputs |
|---|-------|-----------|----------------|--------|---------|
| 1 | Bounding boxes | `bounding_boxes.pbs` | `classifications.bounding_boxes` | Folder of tifs | `*_footprints.gpkg` |
| 2 | Köppen polygon prep | (manual) | — | `World_Koppen.kml`, `AUS_2021_AUST_GDA2020.shp` | `Koppen_Australia_cleaned2.gpkg` |
| 3 | Tile the continent | (manual) | `examples/classifications/demo_bbox_metadata.py` + `demo_barra_bboxs.py` | `Koppen_Australia_cleaned2.gpkg` | `tiles_30000_<State>.gpkg` + per-tile GeoJSONs |
| 4 | Sentinel download | `sentinel.pbs` / `sentinel.sh` | `classifications.sentinel_nci` | `tiles_*.gpkg` | Per-tile Sentinel pickles |
| 5 | Pair Sentinel ↔ tree labels | `merge_inputs_outputs.pbs` / `.sh` | `classifications.merge_inputs_outputs` | Sentinel pickles + tree tifs | Per-tile training CSVs |
| 6 | Combine & split by Köppen | `combine_csvs.pbs` | `classifications.combine_csvs` | Per-tile CSVs | `df_4326_<Koppen>.feather` |
| 7 | Train NN | `neural_network.pbs` | `classifications.neural_network` | Per-Köppen feather | `nn_*.keras`, `scaler_*.pkl` |
| 8 | Batch predict | `predictions.pbs` / `.sh` | `classifications.predictions_batch` | `tiles_*.gpkg` + models | Per-tile predicted tifs |
| 9 | Expand / prep for indices | `prep_expanding.pbs`, `expand_tifs.pbs` / `.sh` | `indices.expand_tifs` | Predicted tifs | Expanded tifs |
| 10 | Merge predicted tifs | `merge_lidar.pbs` / `.sh` | `classifications.merge_tifs` | Folder of predicted tifs | One uint8 tile per sub-region |
| 11 | Shelter indices | `indices.pbs` / `.sh` | `indices.all_indices` | Merged predicted tifs | Shelter / cover / buffer tifs |
| 12 | Merge index outputs | `merge_lidar.pbs` / `.sh` (re-run) | `classifications.merge_tifs` | Index tifs | One continent-scale tif per category |

The `.pbs` filenames are kept as-is for continuity with prior runs even
though `merge_lidar` now runs the renamed `merge_tifs.py` module (and the
"lidar" stage is optional — these scripts merge any folder of tifs).

## Auxiliary scripts

| Script | Purpose |
|--------|---------|
| `elvis.pbs` / `elvis.sh` | Bulk-download LAZ tiles from ELVIS for a GeoJSON grid |
| `lidar.pbs` / `lidar.sh` | Convert downloaded LAZ → canopy-height / binary tree tifs |
| `unzip.pbs` / `unzip.sh` | Decompress LAZ archives in parallel |
| `canopy_height_download.pbs` | Download the Meta Global Canopy Height tiles matching a footprint gpkg |
| `opportunities.pbs` / `.sh` | Run post-hoc opportunity analysis over the shelter categories |
| `distribute.sh`, `distribute_many.sh`, `distribute_undo.sh` | Helpers for fanning work across the xe2 scratch tree |
| `renaming.sh` | One-off filename fixups |
| `cnn.pbs` | Optional CNN model training (alternative to the pixel-wise NN) |

## Launching a stage

All PBS scripts accept their inputs via `qsub -v VAR=value ...`. The bash
launchers iterate over a folder or gpkg and submit one job per unit. For
example:

```bash
cd pbs_scripts
./sentinel.sh                  # one sentinel download job per tile
./merge_lidar.sh               # one merge job per sub-folder
```

All scripts `cd /home/147/cb8590/Projects/shelterbelts/src/shelterbelts/classifications`
and activate the shared conda env at
`/g/data/xe2/cb8590/miniconda/envs/shelterbelts` before invoking Python —
edit those paths if you clone the repo elsewhere.
