# pbs_scripts

This folder contains scripts for running the shelterbelts pipeline at scale. Every .sh file has a matching .pbs file, and every .pbs file has a matching .py file. Some steps may not require a .sh or .pbs file depending on how computationally intensive they are. 

## Workflow

| # | Stage | Script(s) | Inputs | Outputs |
|---|-------|-----------|--------|---------|
| 1 | Bounding boxes | `bounding_boxes.pbs` | Folder of tree tifs (we assume this has already been generated) | `*_footprints.gpkg` |
| 2 | Köppen polygon prep | `demo_koppen_praparation.py` | `World_Koppen.kml`, `AUS_2021_AUST_GDA2020.shp` | `Koppen_Australia_cleaned2.gpkg` |
| 3 | Tiling | `demo_bbox_metadata.py` + `demo_barra_bboxs.py` | `Koppen_Australia_cleaned2.gpkg` | `tiles_30000_<State>.gpkg` + geojsons for ELVIS |
| 4 | Sentinel download | `sentinel.sh`| `tiles_*.gpkg` | Per-tile Sentinel xarrays (and .pkl files if downloading) |
| 5 | Pair Sentinel ↔ tree labels | `merge_inputs_outputs.sh` | Sentinel .pkl's + tree tifs | Per-tile training CSVs |
| 6 | Combine & split by Köppen | `combine_csvs.pbs` | Per-tile CSVs | `df_4326_<Koppen>.feather` |
| 7 | Train NN | `neural_network.pbs` | Per-Koppen feather | `nn_*.keras`, `scaler_*.pkl`, accuracy and loss plots |
| 8 | Batch predict | `predictions.sh` | `classifications.predictions_batch` | `tiles_*.gpkg` + models | Per-tile predicted tifs |
| 9 | Merge predicted tifs | `merge_tifs.sh` | Folder of predicted tifs | Merged predicted tifs |
| 10 | Add border overlaps | `prep_expanding.pbs`, `expand_tifs.sh` | Predicted tifs + Merged predicted tifs | Expanded tifs |
| 11 | Shelter indices | `indices.sh`  | Expanded tifs | Shelterbelt tifs |
| 12 | Merge index outputs | `merge_tifs.sh` (re-run) | Shelterbelt tifs | Merged shelterbelt tifs |

## Auxiliary scripts

| Script | Purpose |
|--------|---------|
| `elvis.sh` | Bulk-download LAZ tiles from ELVIS |
| `unzip.sh` | Decompress LAZ archives in parallel |
| `lidar.sh` | Convert downloaded LAZ → canopy-height / binary tree tifs |
| `canopy_height_download.pbs` | Download the Meta Global Canopy Height tiles matching a footprint gpkg |
| `opportunities.sh` | Next stage in the Indices pipeline |
| `distribute.sh`, `distribute_many.sh`, `distribute_undo.sh` | Helpers for parallelisation |
| `cnn.pbs` | Optional CNN model training instead of the pixel-wise NN |

# Running the scripts

You can launch shell scripts with `./` (e.g. `./sentinel.sh`), and pbs scripts with `qsub` (e.g. `qsub sentinel.pbs`).  

You can run the .py demos as jupyter notebooks on NCI ARE using these settings:
- Sentinel scripts:
    - Module Directories: `/g/data/v10/public/modules/modulefiles`
    - Modules: `dea/20231204`

- Other scripts:
    - Environment base: `/g/data/xe2/cb8590/miniconda`
    - Conda environment: `/g/data/xe2/cb8590/miniconda/envs/shelterbelts`