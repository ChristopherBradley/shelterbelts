# shelterbelts
This repo is for using satellite imagery to map and categorise shelterbelts across Australia, in preparation for measuring impacts on agricultural productivity at scale.

# Google Earth Engine App
You can visualise some results from the repo in this Earth Engine App:  
https://christopher-bradley-phd.projects.earthengine.app/view/shelterbelts  

My slides from the latest Australian National University Research School of Biology (ANU RSB) conference (22 Nov 2025) are here:
https://docs.google.com/presentation/d/1_ItZhrtzTDuZXp-qzwP6mT8-nzrLUdA0sI959rIH5zs/edit?usp=sharing

My poster for the Ecology Society of Australia ESA 2025 is here:
https://docs.google.com/presentation/d/1oC9jB2WT0nFkxfwlVgvFz6CeqZGZr4pwQr8kT_aS2-k/edit?usp=sharing

### Current Methods & Upcoming Plans
The tree predictions come from annual Sentinel-2 imagery largely following a method by Stewart et al. (2025), using a tree/no-tree training dataset provided by Nicolas Pucino.

After the predictions, pixels were categorised using the following method:
- Assign trees from model confidence (50% threshold)
- Assign scattered trees to small groups (< 20 pixels)
- Assign core & buffers to big groups (> 3 pixels thick)
- Assign sheltered vs unsheltered pixels based on percent cover within 100m (10% threshold)
, or wind direction (10 pixels leeward, 5 pixels upwind)
- Assign grassland, cropland, urban and water categories from WorldCover 2021
- Assign riparian and roads trees (3 pixel buffer)
- Assign linear vs non-linear patches by fitting an ellipse and skeleton to each group and applying length and width thresholds (the EE App currently has an outdated version of this)

### Upcoming Plans
- Resolve issue with tree predictions in Western Australia
- Tune thresholds for choosing linear/non-linear/core areas
- Expand shelter categories to the rest of Australia, for each year 2017-2025
- Calculate summary statistics for different regions (states, local govdernment areas, IBRA regions)
- Cleanup demoes and tests and publish in the Journal of Open Source Software
- Include 1m canopy height for all of ACT & NSW
- Analyse effects on productivity & potential future benefits
- Add layers with opportunities for more trees

## Parameter Reference

The main parameters for categorising shelterbelts can be tuned. Below are the default values along with some suggested low and high thresholds:

| Parameter | Default | Low Threshold | High Threshold | Description |
|-----------|---------|---------------|----------------|-------------|
| `min_patch_size` | 20 | 10 | 30 | Minimum pixels to classify as a patch (vs scattered trees) |
| `min_core_size` | 1000 | 100 | 10000 | Minimum pixels to classify as a core area |
| `edge_size` | 3 | 1 | 5 | Distance (pixels) defining the edge region around patch cores |
| `max_gap_size` | 1 | 0 | 2 | Maximum gap (pixels) to bridge when connecting tree clusters |
| `buffer_width` | 3 | 1 | 5 | Distance (pixels) defining buffer zones around features |
| `distance_threshold` | 20 | 10 | 30 | Distance from trees that counts as sheltered (pixels or heights) |
| `density_threshold` | 5 | 3 | 10 | Minimum percentage tree cover that counts as sheltered |
| `wind_threshold` | 20 | 10 | 30 | Wind speed (km/h) used to determine dominant wind direction |
| `wind_method` | WINDWARD | MOST_COMMON | ALL | Method to determine primary wind direction |
| `strict_core_area` | strict | non-strict | strict | Whether to enforce strict connectivity for core areas |
| `min_shelterbelt_length` | 20 | 10 | 30 | Minimum length to classify as a shelterbelt |
| `max_shelterbelt_width` | 6 | 4 | 8 | Maximum width to classify as a shelterbelt |

Parameters can be modified when calling functions directly in Python or via command-line arguments. For example:

```bash
python -m shelterbelts.indices.tree_categories input.tif --min_patch_size 30 --edge_size 5
```

# Setup

## Local Setup
1. Download and install Miniconda from https://www.anaconda.com/download/success
2. Add the miniconda filepath to your ~/.zhrc, e.g. export PATH="/opt/miniconda3/bin:$PATH"
3. `git clone https://github.com/ChristopherBradley/shelterbelts.git`
4. `cd shelterbelts`
5. `conda env create -f environment.yml`
6. `conda activate shelterbelts`

## Setup on gadi at the National Computing Infrastructure (NCI)
1. [Create an account](https://my.nci.org.au/mancini/login) and request access to the projects xe2 (Borevitz Lab), v10 (Digital Earth Australia modules), ka08 (Sentinel-2 Imagery), ob53 (BARRA Wind).
2. `ssh {username}@nci.org.au` and enter the password used to create your account.
3. `git clone https://github.com/ChristopherBradley/shelterbelts.git`
4. There are examples usage of the environments in pbs_scripts

## Usage on NCI ARE (National Computing Infrastructure's Australian Research Environment)
1. Login here: https://are.nci.org.au/
2. Go to JupyterLab and create a session with 1 hour, queue normalbw, compute size small, project xe2, storage gdata/+gdata/xe2+gdata/v10+gdata/ka08+gdata/ob53, python environment base /g/data/xe2/cb8590/miniconda, conda environment /g/data/xe2/cb8590/miniconda/envs/shelterbelts. Alternatively, can use Module Dircetories /g/data/v10/public/modules/modulefiles and Modules: dea/20231204.
3. Right click any .py file and open as a jupyter notebook. Currently some debugging arguments are usually commented out at the bottom of each file. I'm planning to move these to tests/demos.

# Examples
There are jupyter notebooks to demo the functionality of this repo in `notebooks`. Also, there are .pbs scripts for submitting synchronous jobs to gadi in `pbs_scripts`, along with .sh scripts to submit many jobs in parallel. The main python files are in `src/shelterbelts` and these can all be run from the command line as well. The `tests` have the same examples as `notebooks` but are more convenient to run all at once (but less convenient for demo-ing/understanding the functionality).  

# Testing
From the project root:
`qsub -I -P xe2 -q copyq -l ncpus=1 -l mem=8GB -l walltime=02:00:00 -l storage=gdata/xe2+scratch/xe2 -l wd`
`conda activate /g/data/xe2/cb8590/miniconda/envs/shelterbelts`
`pytest tests/test_indices`

