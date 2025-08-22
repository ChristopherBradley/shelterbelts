install.packages("lidR")
install.packages("RCSF")
install.packages("sp")
install.packages("raster")

library(lidR)
library(raster)
library(sf)

# Downloaded the 6 .laz files from ELVIS that overlapped with Esdale
folder <- "/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/"
laz_files <- list.files(folder, pattern = "\\.laz$", full.names = TRUE)

rasters_list <- list()
for (laz_file in laz_files) {
  print(paste("Working on: ", laz_file))
  las <- readLAS(laz_file)

  # Binary tree raster: Classification code 5 means points > 2m. Documentation is here: https://www.spatial.nsw.gov.au/__data/assets/pdf_file/0004/218992/Elevation_Data_Product_Specs.pdf 
  tree_raster10 <- grid_metrics(las, ~ as.numeric(any(Classification %in% c(5))), res = 10)
  rasters_list[[length(rasters_list)+1]] <- tree_raster10
}

# Merge all rasters into one
merged_raster <- do.call(merge, rasters_list)
writeRaster(merged_raster, "~/Desktop/ESDALE_medium_vegetation.tif", overwrite = TRUE)


###############

# Trying out the dalponte method on a canopy height model created in python with pdal
las <- readLAS("/Users/christopherbradley/Documents/PHD/Data/ESDALE/NSW_LiDAR_2018_80cm/Point Clouds/AHD/Brindabella201802-LID2-C3-AHD_6746112_55_0002_0002.laz")
chm <- raster("/Users/christopherbradley/repos/PHD/shelterbelts/src/shelterbelts/classifications/chm.tif")
chm <- projectRaster(chm, crs = crs(las))
ttops <- locate_trees(chm, lmf(ws = 5))
system.time({
crowns <- segment_trees(las, dalponte2016(chm, ttops))
})

# Convert segmented crowns to polygons
crowns_delineated <- delineate_crowns(crowns)

# Convert to sf object for easy manipulation and export
crowns_sf <- st_as_sf(crowns_delineated)

# Save polygons as GeoPackage
filename <- "~/Desktop/tree_crowns.gpkg"
st_write(crowns_sf, filename, driver = "GPKG", delete_layer = TRUE)

