# pbs_scripts

This folder contains scripts for running the shelterbelts pipeline at scale. Every .sh file has a matching .pbs file, and every .pbs file has a matching .py file. Some steps may not require a .sh or .pbs file depending on how computationally intensive they are. 

## Workflow

| # | Stage | Script(s) | Inputs | Outputs |
|---|-------|-----------|--------|---------|
| 1 | Bounding boxes | `bounding_boxes.pbs` | Folder of tree tifs (we assume this has already been generated) | `*_footprints.gpkg` |
| 2 | Köppen polygon prep | `demo_koppen_praparation.py` | `World_Koppen.kml`, `AUS_2021_AUST_GDA2020.shp` | `Koppen_Australia_cleaned2.gpkg` |
| 3 | Tiling | `analysis/barra_bboxs.py` (+ `analysis/split_ag_bboxs.py` for the ag-only option) | `barra_bboxs_full.gpkg` | `barra_bboxs_aus.gpkg` / `barra_bboxs_ag.gpkg` + chunked `BARRA_bboxs_*/*.gpkg` for `predictions.sh` |
| 4 | Sentinel download | `sentinel.sh`| `tiles_*.gpkg` | Per-tile Sentinel xarrays (and .pkl files if downloading) |
| 5 | Pair Sentinel ↔ tree labels | `merge_inputs_outputs.sh` | Sentinel .pkl's + tree tifs | Per-tile training CSVs |
| 6 | Combine & split by Köppen | `combine_csvs.pbs` | Per-tile CSVs | `df_4326_<Koppen>.feather` |
| 7 | Train NN | `neural_network.pbs` | Per-Koppen feather | `nn_*.keras`, `scaler_*.pkl`, accuracy and loss plots |
| 8 | Batch predict | `predictions.sh` | `tiles_*.gpkg` + models | Folder of predicted tifs |
| 9 | Bin predictions into regions | `prep_expanding.pbs` | Folder of predicted tifs | `subfolders/lat_X_lon_Y/` per-region subfolders |
| 10 | Merge predicted tifs | `merge_tifs.sh` | `subfolders/lat_X_lon_Y/` (per region) | `subfolders/lat_X_lon_Y_merged_predicted.tif` (one per region) |
| 11 | Bounding boxes for merged tifs | `bounding_boxes.pbs` | `subfolders/*_merged_predicted.tif` | `subfolders/subfolders__footprints.gpkg` |
| 12 | Add border overlaps | `expand_tifs.sh` | Per-region raw tiles + merged tifs + footprints gpkg | `expanded/lat_X_lon_Y/*_expanded<N>.tif` |
| 13 | Shelter indices | `indices.sh`  | Expanded tifs | Shelterbelt tifs |
| 14 | Merge index outputs | `merge_tifs.sh` (re-run) | Shelterbelt tifs | Merged shelterbelt tifs |

### Compute cost (all ~133k 4 km tiles / 148 regions across agricultural Australia)

- **Step 8** (Batch predict) costs **~5 KSU per year** for all agricultural regions in Australia.
- **Steps 9-12** (bin → merge → footprints → expand) cost **~50 SU total**.
- **Step 13** (Shelter indices) costs **~0.5 KSU per method** with the default wind method (~0.004 SU per 4 km tile × ~133k tiles; measured on the 2025 ag run). The `more_windmethod` (ANY = 8 directions) is ~2× (**~1.2 KSU**); the percent methods are cheaper (**~0.3-0.5 KSU**). The 2025 six-method run was therefore **~3-4 KSU total**.
- **Step 14** (Merge index outputs) costs only **tens of SU** (one short merge job per ~200 km region).
- For reference, the 2026 **Global Canopy Height v2** variant (10 m percent-cover binning → height-aware WINDWARD indices + opportunities → merge → Australia-wide value_counts) cost **~0.66 KSU total** for all ag regions: ~0.06 KSU binning, ~0.58 KSU indices+merge, the remainder value_counts + ag-boundary masking. All jobs ran at 4 GB / 1 CPU (peak ~2.7 GB).

## Auxiliary scripts

| Script | Purpose |
|--------|---------|
| `elvis.sh` | Bulk-download LAZ tiles from ELVIS |
| `unzip.sh` | Decompress LAZ archives in parallel |
| `lidar.sh` | Convert downloaded LAZ → canopy-height / binary tree tifs |
| `canopy_height_download.pbs` | Download the Meta Global Canopy Height tiles matching a footprint gpkg |
| `opportunities.sh` | Next stage in the Indices pipeline |
| `distribute.sh`, `distribute_many.sh`, `distribute_undo.sh` | Helpers for parallelisation |

# Running the scripts

You can launch shell scripts with `./` (e.g. `./sentinel.sh`), and pbs scripts with `qsub` (e.g. `qsub sentinel.pbs`).  

You can run the .py demos as jupyter notebooks on NCI ARE using these settings:
- Sentinel scripts:
    - Module Directories: `/g/data/v10/public/modules/modulefiles`
    - Modules: `dea/20231204`

- Other scripts:
    - Environment base: `/g/data/xe2/cb8590/miniconda`
    - Conda environment: `/g/data/xe2/cb8590/miniconda/envs/shelterbelts`