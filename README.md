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

I was in a bit of a rush before ESA, but I'm planning to clean these up and prepare a publication for the Journal of Open Source Software (JOSS) before the end of 2025...
