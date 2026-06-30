---
title: 'Shelterbelts: A Python library for characterising farmland tree groups and their influence on nearby pasture and cropland'
tags:
  - Python
  - remote sensing
  - agriculture
  - agroforestry
  - shelterbelts
  - windbreaks
  - Sentinel-2
  - Australia
authors:
  - name: Christopher Bradley
    orcid: 0009-0008-2291-8433
    affiliation: 1
    corresponding: true
  - name: John T. Burley
    orcid: 0000-0003-4702-5056
    affiliation: 1
  - name: Justin Borevitz
    orcid: 0000-0001-8408-3699
    affiliation: 1
affiliations:
  - name: Australian National University, Canberra, Australia
    index: 1
date: 30 June 2026
bibliography: paper.bib
---

# Summary

Shelterbelts is an open source Python library for mapping and categorising groups of trees in agricultural landscapes, and the corresponding shelter provided to nearby pixels based on the wind direction or density of tree cover. Starting from a binary tree raster, pixels are labelled into ecologically meaningful categories based on their structure and location including: scattered trees, forest core and edge, riparian and road buffers, and other linear (windbreaks/tree-rows) and non-linear patches. The library also provides functions to fetch and transform raster datasets for wind speed and direction, elevation and topographic attributes, along with national road and hydroline vector datasets. You can also use the library to process LiDAR through crown delineation, or fetch and transform global canopy height and land cover data, or train a model to predict tree cover from satellite imagery, to create the necessary input binary raster for the pipeline. The software was designed and tested on landscapes in Australia, but can be applied anywhere. It can be run locally in jupyter notebooks or at scale in high-throughput environments such as the National Computing Infrastructure (NCI). The github repository is available at <https://github.com/ChristopherBradley/shelterbelts>. Example classifications for the whole of Australia can be viewed at this Earth Engine App <https://christopher-bradley-phd.projects.earthengine.app/view/shelterbelts>.

# Statement of Need

Shelterbelts - also known as windbreaks or tree rows - can provide benefits for biodiversity while also helping mitigate climate change through carbon drawdown, and at the same time increasing nearby crop, pasture and animal production [@ewer2023; @udawatta2022]. These production benefits may happen indirectly due to reduced erosion from wind and water, reduced salinity and nutrient runoff, increased soil fertility from carbon sequestration, and increased pollinator abundance and beneficial insect predators [@reid2018]. Strong gusts or chronic winds can also alter plant growth habits (for example with smaller leaves, shorter and wider stems, and less fruits and seeds), and cause damage through lodging, abrasion, and leaf tearing or folding, all of which can cause severe water loss, entry of pathogens, or even wilting and death [@gardiner2016].

Due to this multitude of mechanisms, the magnitude of benefits from shelterbelts is highly variable depending on factors such as climate, topography, crop type and wind regime [@cleugh2002; @baker2025]. Similarly, the magnitude of biodiversity benefits depends on factors such as patch sizes and connectivity, as well as water availability due to the topographic position [@lindenmayer2009a; @lindenmayer2009b]. This variability means it is hard to quantify the benefits of existing trees and new plantings at specific sites, reducing the likelihood of adoption [@powell2009].

Another factor that may contribute to a lack of adoption is the complexity in choosing where to plant new shelterbelts [@fleming2019]. Some guidelines suggest planting along contours [@greeningaustralia2014], ridges [@watson2019], gullies [@wentworth2024], roadsides [@new2021], or anywhere to achieve a certain density of tree cover [@bird1992]. Regardless of where they go, there is a clear desire to add large numbers of trees to farmland landscapes for carbon drawdown and resilience to future climate extremes [@daff2025], so new tools to help plan where these should go are urgently needed [@nsw2022].

Improvements in satellite imagery over the last decade means there is potential to map variability of productivity due to shelterbelts at scale, similar to work done for conservation tillage [@cambron2024] and crop rotations [@kluger2025]. But a lack of available tools has led to studies like @liu2022 using more manual methods such as the linear ruler tools in Google Earth Pro, which limits the scale of analyses that can be done. There is a clear need for tools that can create ecologically and agriculturally relevant data about farmland trees and shelter to better understand their current distributions, benefits, and opportunities.

# State of the Field

There exist many described methods in the literature that helped inspire aspects of the shelterbelts software, but often the code is proprietary or simply not released. Land cover mapping and canopy height modelling is an active area of research, meaning that global vegetation datasets are continually improving [@potapov2021; @zanaga2022; @lang2023; @tolan2024; @brandt2026]. However these kinds of studies generally do not look further at the spatial arrangement to characterise tree groups.

Moving past regular tree-cover, @wolstenholme2025 used aerial imagery to map hedgerows in a county in the UK, and then used the 'land' and 'roads, tracks and paths' labels from the UK ordnance survey master map to help identify gaps in hedgerow. At a larger scale, @muro2025 used PlanetScope satellite imagery to map hedgerows across Germany and calculated the centerline and length of all polygons. However, they did not describe how the rasters were converted to polygons or release any code or datasets regarding the polygons.

@hopkins2025 also used aerial imagery to map windbreaks in 35 counties in the United States using ArcGIS Pro. They used the Straight and Narrow Feature Index and Windbreak Sinuosity Index demonstrated by @liknes2017 which looks for a specific shape like a north-south or east-west straight line in a raster. This was developed as an alternative to a method proposed by @aksoy2010 which skeletonizes features and checks the width along the skeleton to define windbreaks. Later, @deng2023 proposed another method for identifying shelterbelts that finds the endpoints of each feature and joins the centerlines, to connect shelterbelts with gaps in them. @hopkins2025 also proposed incorporating vector data on streams, rivers, aqueducts, and canals, lakes, wetlands, and ponds to distinguish riparian buffers from other woody features, but did not implement this.

Instead of a pixel-based approach, @wiseman2009 applied object-oriented image analysis on aerial imagery using the commercial software Definiens Professional, to identify shelterbelts in the North Cypress municipality in Canada, along with attributes including the length and width of each shelterbelt. @thompson2023 used the same commercial software (now renamed Trimble eCognition) to identify hedgerows and windbreaks from aerial imagery in 3 counties in California, and find adoption hotspots. @yang2021 also used eCognition to do object segmentation on satellite imagery in a roughly 20km square region in North East China, to identify shelterbelts and estimate widths. Then they used linear regression to estimate porosity and height, and calculated euclidean distance in ArcGIS to estimate wind reduction nearby. Similarly, @stewart2024 described methods for modelling changes in crop and pasture productivity, which they implemented using the sf and terra packages in R drawing on research later published by @baker2025, but did not release the code to apply this to new locations.

Similar to the final stage of the shelterbelts pipeline, there are a number of landscape ecology libraries that calculate patch and class metrics such as FragStats [@mcgarigal2023], landscapemetrics [@hesselbarth2019] and PyLandStats [@bosch2019]. FragStats is a proprietary application written in Visual C++ whereas landscapemetrics and PyLandStats are both open source (written in R and Python respectively), however none of these libraries produce shelterbelt specific metrics such as the width, length and orientation.

# Software Design

## Workflows

Shelterbelts has been designed to be modular and adaptable, supporting different priorities and assumptions about how tree configurations matter to the landscape. It supports six main workflows in increasing complexity:

1. Browse the pre-computed outputs in the Earth Engine App from default parameters.
2. Use the indices_latlon function to auto-download the global canopy height model (Meta/Tolan) and generate shelter categories and metrics for a specific location.
3. Download your own tree raster from another source and use the indices_tif function to generate shelter categories and metrics for that raster.
4. Download a laz file and use the lidar function to generate a tree raster, before using the indices_tif function to generate shelter categories and metrics for that raster.
5. Use sentinel_dea.py to download sentinel imagery and apply the bundled neural network model to generate a tree raster, before applying the indices_tif function. A sensitivity analysis of the bundled model in different shelter categories and distances from trees is planned for a companion paper.
6. Use sentinel_nci and a folder of tree tif files to train your own model that predicts tree rasters, then apply this to a new location and pass the output into the indices_tif function.

![Flow chart of usage and underlying design of the shelterbelts repository.\label{fig:flowchart}](flowchart.png)

![Example geolocated output tif files at each stage of the pipeline.\label{fig:pipeline}](pipeline.png)

## Principles

Each source file has a corresponding Jupyter Notebook to demonstrate usage and provide a familiar environment for new users to play around with the outputs. The Jupyter Notebooks are initially saved as .py files during development using the Jupytext library to avoid overloading the git history with slight changes to large image files. These are committed as a .ipynb file once finalised so the outputs are easily viewable from within the github interface.

Each source file also has a corresponding test file for common scenarios to quickly check that new changes do not break existing parts of the codebase. We have prioritised testing common scenarios and past known failure mechanisms over test completeness to better support fast iterative development.

Documentation is auto-generated by sphinx with doctests and plot_directives to demonstrate usage and example outputs, similar to the Jupyter Notebooks. These docs are hosted with GitHub Pages, meaning they can be easily generated locally with no additional costs from cloud computing since the doctests take a while to run.

Each stage of the pipeline has the option to generate an intermediate file for further analysis, or just return the data to pass to the next stage of the pipeline (reducing read/write overhead and improving performance). These intermediate files are stored as georeferenced tif files, usually with the uint8 datatype and an embedded colour map for easy viewing in QGIS. When running at scale, the canopy height or percent tree cover rasters can also be converted to the uint8 datatype since the uncertainty in these datasets is generally higher than 1m height or 1% cover, and this greatly reduces the compute and storage costs.

Each source file has a command line interface to support running in a high-throughout environment such as the National Computing Infrastructure (NCI). There are example shell and pbs scripts in the repository that were used to generate the datasets in the public Earth Engine App, although these are not packaged as part of the source code.

# Outlook for Research and Development

The shelterbelts pipeline has already been run across all agricultural regions in Australia at 10m resolution using six different configurations of input parameters - a default, high and a low threshold for the methods based on wind direction, and based on percentage tree cover. These maps are publicly available in an Earth Engine app [@gorelick2017], with the ability to choose a region of interest and download any layer as a geotif file for further analysis. Work is currently underway to generate annual maps of these categories, and regional summary statistics in Australia's bioregions, GRDC cropping zones, and local government areas.

The Earth Engine app allows you to toggle the global canopy height and worldcover layers, and you can use these data sources from within the repository to apply the tree and shelter categorisation at any location globally. Functions are provided for creating gully rasters from global elevation data and road rasters from Open Street Maps, used as inputs to the pipeline. Functions are not currently provided for downloading global wind direction data, but these do exist (e.g. ERA-5), or the pipeline can be run without wind data by using the percent cover or any direction wind methods.

There is growing policy and public funding support for planting trees on farms for carbon drawdown, biodiversity, and agricultural resilience [@daff2025; @nsw2022], but still large uncertainty about site specific benefits which may limit adoption. The shelter categories and per-pixel degree of shelter generated by this repository can be a direct input into analyses that compare this with satellite derived productivity, providing the ability to map this variability at scale.

# Acknowledgements

This work used computational resources provided by the National Computational Infrastructure (NCI Australia). We thank Geoscience Australia, Digital Earth Australia, and the developers of the open-source software used in this work.

# AI Usage Disclosure

Various versions of Claude Code were used to assist with software development, particularly debugging and proofreading.

# References
