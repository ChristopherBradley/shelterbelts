# shelterbelts
Using satellite imagery for mapping and categorising shelterbelts across Australia, in preparation for measuring impacts on agricultural productivity at scale.

# Google Earth Engine App
You can visualise some results from the repo in this Earth Engine App:
https://christopher-bradley-phd.projects.earthengine.app/view/testingexampleshelterbelts
(will update to this link soon: https://christopher-bradley-phd.projects.earthengine.app/view/shelterbelts)

# Local Setup
1. Download and install Miniconda from https://www.anaconda.com/download/success
2. Add the miniconda filepath to your ~/.zhrc, e.g. export PATH="/opt/miniconda3/bin:$PATH"
3. `git clone https://github.com/ChristopherBradley/shelterbelts.git`
4. `cd shelterbelts`
5. `conda env create -f environment.yml`
6. `conda activate shelterbelts`

# Setup on gadi at the National Computing Infrastructure (NCI)
1. [Create an account](https://my.nci.org.au/mancini/login) and request access to the projects xe2 (Borevitz Lab), v10 (Digital Earth Australia modules), ka08 (Sentinel-2 Imagery), ob53 (BARRA Wind).
2. `ssh {username}@nci.org.au` and enter the password used to create your account.
3. `git clone https://github.com/ChristopherBradley/shelterbelts.git`
4. There are examples usage of the environments in pbs_scripts

# Usage on NCI ARE (National Computing Infrastructure's Australian Research Environment)
1. Login here: https://are.nci.org.au/
2. Go to JupyterLab and create a session with 1 hour, queue normalbw, compute size small, project xe2, storage gdata/+gdata/xe2+gdata/v10+gdata/ka08+gdata/ob53, python environment base /g/data/xe2/cb8590/miniconda, conda environment /g/data/xe2/cb8590/miniconda/envs/shelterbelts. Alternatively, can use Module Dircetories /g/data/v10/public/modules/modulefiles and Modules: dea/20231204.
3. Right click any .py file and open as a jupyter notebook. Currently some debugging arguments are usually commented out at the bottom of each file. I'm planning to move these to tests/demos.

# Methods and Parameters
There are jupyter notebooks to demo the functionality of this repo in `notebooks`. Also, there are .pbs scripts for submitting synchronous jobs to gadi in `pbs_scripts`, along with .sh scripts to submit many jobs in parallel. The main python files are in `src/shelterbelts` and these can all be run from the command line as well. The `tests` have the same examples as `notebooks` but are more convenient to run all at once (but less convenient for demo-ing/understanding the functionality).


# Running at scale

Using Elvis
- elvis_geojson.py
- 

Starting from a .laz file (1m CHM)
- 

Starting from a folder of binary tif files (classifications)
- 

Starting from a folder of percent cover files (indices)
- predictions
- prep_exanding
- merging
- expansion
- indices