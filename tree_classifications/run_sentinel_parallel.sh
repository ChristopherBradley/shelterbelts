#!/bin/bash
#PBS -N Sentinel
#PBS -l mem=32GB
#PBS -l ncpus=8
#PBS -l jobfs=24GB
#PBS -P xe2
#PBS -l walltime=00:15:00
#PBS -l storage=gdata/xe2+gdata/v10+gdata/ka08
#PBS -q normal

wd=/home/147/cb8590/Projects/shelterbelts/tree_classifications 
indir='/g/data/xe2/cb8590/Nick_Aus_treecover_10m'
outdir='/scratch/xe2/cb8590/Nick_sentinel'

# Print out input variables to the error log
echo "Running with the following input variables:"
echo "indir: $indir"
echo "tif: $tif"
echo "year: $year"
echo "outdir: $outdir"
echo "-------------------"

# Requirements:
# Needs access to project v10 to load the dea modules
# (also ka08 and xe2)

#cd /home/106/jb5097/Projects/PaddockTS
cd $wd

# Setup DEA environment modules for running the Python script
module use /g/data/v10/public/modules/modulefiles
module load dea/20231204

python3 sentinel_download.py --indir $indir  --tif $tif --year $year --outdir $outdir