import glob
import math
import pandas as pd
import subprocess

# Change directory to this repo
import sys, os
repo_name = "shelterbelts"
if os.path.expanduser("~").startswith("/home/"):  # Running on Gadi
    repo_dir = os.path.join(os.path.expanduser("~"), f"Projects/{repo_name}")
elif os.path.basename(os.getcwd()) != repo_name:
    repo_dir = os.path.dirname(os.getcwd())  # Running in a jupyter notebook 
else:  # Already running locally from repo root
    repo_dir = os.getcwd()
os.chdir(repo_dir)
sys.path.append(repo_dir)
print(f"Running from {repo_dir}")

# Load the filenames for all the sentinel tiles I've downloaded
sentinel_dir = "/scratch/xe2/cb8590/Nick_sentinel"
tiles = glob.glob(f'{sentinel_dir}/*.pkl')
print(len(tiles))

rows = tiles[:16]
workers = 4
batch_size = math.ceil(len(rows) / workers)
batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
print("num_batches: ", len(batches))
print("num tiles in first batch", len(batches[0]))

# Save the tiles in each batch in a csv file so I can launch a subprocess that works on each batch
batch_files = []
for i, batch in enumerate(batches):
    batch_file = f"/g/data/xe2/cb8590/models/batches/batch_{i}.csv"
    df = pd.DataFrame(batch)
    df.to_csv(batch_file, index=False)
    print("Saved", batch_file)
    batch_files.append(batch_file)

# +
# %%time
procs = []
for batch_file in batch_files:
    # p = subprocess.Popen(["python", "tree_classifications/predictions_batch.py", "--input", batch_file])
    p = subprocess.Popen(["python", "tree_classifications/predictions_batch.py"])
    procs.append(p)

# Wait for all to finish
for p in procs:
    p.wait()
    
# Excellent, this actually happened in parallel
# -

# This happens in series so no point
# %%time
for batch_file in batch_files:
    subprocess.run(["python", "tree_classifications/predictions_batch.py"])
        # subprocess.run(["python", "predictions_batch.py", "--input", batch_file])


