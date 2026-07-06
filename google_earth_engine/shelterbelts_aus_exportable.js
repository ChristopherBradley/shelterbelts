
Map.setOptions('SATELLITE');
Map.drawingTools().setShown(false);
Map.drawingTools().setDrawModes(['rectangle']);

var greenPalette = [
  '#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d',
  '#238b45','#006d2c','#00441b'
];

var shelterPalette = {
  0:   [255, 255, 255],  // White: Outside region
  10:  [0, 100, 0],      // Green: Worldcover tree cover
  11:  [122, 82, 0],     // Brown: Scattered Trees
  12:  [8, 79, 0],       // Dark green: Patch core
  13:  [14, 138, 0],     // Medium green: Patch edge
  14:  [22, 212, 0],     // Bright green: Other trees
  15:  [29, 153, 105],   // Bluey green: Trees in gullies
  16:  [127, 168, 57],   // Orange: Trees on ridges
  17:  [129, 146, 124],  // Silver: Trees next to roads
  
  18: [190, 160, 60],  // Light brown: Linear patches
  19: [165, 195, 45],  // Bright olive green: Non-linear patches

  20:  [255, 187, 34],   // Orange: Shrubs
  30:  [255, 255, 76],   // Yellow: Worldcover grassland
  31:  [203, 219, 115],  // Dull yellow: unsheltered grassland
  32:  [255, 255, 76],   // Bright yellow: sheltered grassland
  40:  [240, 150, 255],  // Pink: Worldcover cropland
  41:  [146, 104, 143],  // Dull pink: unsheltered cropland
  42:  [240, 150, 255],  // bright pink: sheltered cropland
  50:  [250, 0, 0],      // Red: Built-up
  60:  [180, 180, 180],  // Grey: Bare
  70:  [240, 240, 240],  // White: Snow
  80:  [0, 100, 200],    // Blue: Water
  90:  [0, 150, 160],    // Worldcover wetland
  95:  [0, 207, 117],    // Worldcover mangroves
  100: [250, 230, 160],  // Worldcover moss and Lichen
};

var fullyTransparentClasses = [0, 31, 41];
var partialTransparentClasses = [32, 42];
var shelteredTransparency = 0.3;

var exportLayers = {};

// Function for plotting shelter categories with some transparency
function styleShelterImage(img, layerName, checkbox, transparency) {
   // Normalize ImageCollection to Image
  if (img instanceof ee.ImageCollection) {
    img = img.mosaic();
  }
  
  // Remove fully transparent classes
  var classKeys = Object.keys(shelterPalette)
    .map(function(k){ return parseInt(k, 10); })
    .filter(function(k){ return fullyTransparentClasses.indexOf(k) === -1; })
    .sort(function(a,b){ return a - b; });
    
  // Create a consecutive colour scheme
  var paletteHex = classKeys.map(function(k){
    var rgb = shelterPalette[k];
    return '#' + rgb.map(function(c){
      var h = c.toString(16);
      return (h.length === 1) ? '0' + h : h;
    }).join('');
  });
  var targetIndices = classKeys.map(function(_, i){ return i; });
  var bandName = img.bandNames().get(0);
  var classesImg = img.select([bandName]);
  var remapped = classesImg.remap(classKeys, targetIndices, -1);
    var baseMask = remapped.neq(-1);
  
  // Add partial transparency for some classes
  var mask = baseMask;
  partialTransparentClasses.forEach(function(cls) {
    mask = mask.where(classesImg.eq(cls), shelteredTransparency);
  });
  var styled = remapped.updateMask(mask);
  
  // Add to the map
  Map.addLayer(styled,
               {min: 0, max: targetIndices.length - 1, palette: paletteHex},
               layerName, checkbox, transparency);
  
  return styled;
}

// Map.centerObject(image, 10);
Map.setCenter(148.471268, -34.389131, 12);  // (lon, lat, zoom)


///////////////////////////////////////////////////////////
// WorldCover 2020
var wcClasses = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100];
var wcPalette = ['006400', 'ffbb22', 'ffff4c', 'f096ff', 'fa0000',
                 'b4b4b4', 'f0f0f0', '0064c8', '0096a0', '00cf75', 'fae6a0'];
var wcIndices = wcClasses.map(function(_, i) { return i; });
var wc = ee.ImageCollection('ESA/WorldCover/v100').first().select('Map');
var wcRemapped = wc.remap(wcClasses, wcIndices, -1);
Map.addLayer(
  wcRemapped.updateMask(wcRemapped.neq(-1)),
  {min: 0, max: wcClasses.length - 1, palette: wcPalette},
  'WorldCover 2020', false, 1
);

///////////////////////////////////////////////////////////
// Canopy Height Model v2 (Meta & WRI)
var viridis = ['440154','482878','3e4989','31688e','26828e','1f9e89','35b779','6ece58','b5de2b','fde725'];
var chm = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight').mosaic();
Map.addLayer(
  chm.updateMask(chm.gt(0)),
  {min: 0, max: 25, palette: viridis},
  'Meta Canopy Height v2', false, 0.8
);

///////////////////////////////////////////////////////////
// Aus trees
var aus2017 = ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_2017_noxy_predictions_lat8-18'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_2017_noxy_predictions_lat20-24'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_2017_noxy_predictions_lat26-28'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_2017_noxy_predictions_lat30-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_2017_noxy_predictions_lat34-42')
]).mosaic();
Map.addLayer(aus2017.updateMask(aus2017.gt(50)), {min: 50, max: 100, palette: ['00FF00']}, '2017 tree confidence 50%', false, 0.65);
Map.addLayer(aus2017.updateMask(aus2017.gt(90)), {min: 90, max: 100, palette: ['00FF00']}, '2017 tree confidence 90%', false, 0.65);

// I combined some of these into image collections manually with drag dropping, before deciding later it was easier to this in the app code.
var aus2020 = ee.ImageCollection('projects/ee-christopher-bradley/assets/Aus_2020_noxy_predictions').mosaic();
Map.addLayer(aus2020.updateMask(aus2020.gt(50)), {min: 50, max: 100, palette: ['00FF00']}, '2020 tree confidence 50%', false, 0.65);
Map.addLayer(aus2020.updateMask(aus2020.gt(90)), {min: 90, max: 100, palette: ['00FF00']}, '2020 tree confidence 90%', false, 0.65);

var aus2024 = ee.ImageCollection('projects/ee-christopher-bradley/assets/Aus2024_noxy_predictions').mosaic();
Map.addLayer(aus2024.updateMask(aus2024.gt(50)), {min: 50, max: 100, palette: ['00FF00']}, '2024 tree confidence 50%', false, 0.65);
Map.addLayer(aus2024.updateMask(aus2024.gt(90)), {min: 90, max: 100, palette: ['00FF00']}, '2024 tree confidence 90%', false, 0.65);


///////////////////////////////////////////////////////////
// Shelter classifications
var moreDensityImg = ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-percentmethod_lat10-26'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-percentmethod_lat28-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-percentmethod_lat34-42')
]).mosaic();
styleShelterImage(moreDensityImg, '2020 more density method', false);

var lessDensityImg = ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-percentmethod_lat10-26'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-percentmethod_lat28-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-percentmethod_lat34-42')
]).mosaic();
styleShelterImage(lessDensityImg, '2020 less density method', false);

var defaultDensityImg = ee.ImageCollection('projects/ee-christopher-bradley/assets/Aus_ag2020_default-percentmethod').mosaic();
styleShelterImage(defaultDensityImg, '2020 default density method', false);

var moreWindImg = ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-windmethod_lat10-28'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-windmethod_lat28-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-windmethod_lat34-42')
]).mosaic();
styleShelterImage(moreWindImg, '2020 more wind method', false);

var lessWindImg = ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-windmethod_lat10-26'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-windmethod_lat28-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-windmethod_lat34-42')
]).mosaic();
styleShelterImage(lessWindImg, '2020 less wind method', false);

var defaultWindImg = ee.ImageCollection('projects/ee-christopher-bradley/assets/Aus_ag2020_default-windmethod').mosaic();
styleShelterImage(defaultWindImg, '2020 default wind method', true);


//////////////////////////////////////////////////////////
// Create an info panel
var infoPanel = ui.Panel({
  style: {
    width: '270px',
    position: 'top-right',
    padding: '4px',
    backgroundColor: 'white'
  }
});

// Title
infoPanel.add(ui.Label({
  value: 'Shelterbelts',
  style: {fontWeight: 'bold', fontSize: '14px', margin: '0 0 2px 0'}
}));

infoPanel.add(ui.Label(
  'This is a work-in-progress for visualising shelterbelt categories. Code and more details are available here:'
    , {whiteSpace: 'pre-line', fontSize: '11px', margin: '0 0 0px 0'}
));

infoPanel.add(ui.Label(
  'https://github.com/ChristopherBradley/shelterbelts',
  {fontSize: '11px', color: 'blue', textDecoration: 'underline', margin: '0 0 6px 0'},
  'https://github.com/ChristopherBradley/shelterbelts'
));
infoPanel.add(ui.Label(
  'Feedback appreciated! \nchristopher.bradley@anu.edu.au\n'
    , {whiteSpace: 'pre-line', fontSize: '11px', margin: '0 0 0px 0'}
));

// Add the panel to the map
Map.add(infoPanel);



//////////////////////////////////////////////////////////////////////
// // Adding a legend. 
var classLabels = {
  0: 'Not Trees',
  10: 'Tree cover',
  11: 'Scattered Trees',
  12: 'Patch Core',
  13: 'Patch Edge',
  14: 'Other Trees',
  15: 'Trees in Gullies',
  16: 'Trees on Ridges',
  17: 'Trees next to Roads',
  18: 'Linear Patches',
  19: 'Non-linear Patches',
  20: 'Shrubland',
  30: 'Grassland',
  31: 'Unsheltered Grassland',
  32: 'Sheltered Grassland',
  40: 'Cropland',
  41: 'Unsheltered Cropland',
  42: 'Sheltered Cropland',
  50: 'Built-up',
  60: 'Bare',
  70: 'Snow and ice',
  80: 'Permanent water bodies',
  90: 'Herbaceous wetland',
  95: 'Mangroves',
  100: 'Moss and lichen'
};

// // Manually choosing values, because there are some that don't show up enough to be worthwhile.
var presentClasses = [11, 12, 13, 15, 17, 18, 19, 32, 42, 50, 80, 20, 60];

var legend = ui.Panel({
  style: {
    position: 'top-right',
    padding: '4px 8px',
    maxWidth: '200px',
    maxHeight: '40vh'
  }
});

legend.add(ui.Label({
  value: 'Shelter Categories',
  style: {
    fontWeight: 'bold',
    fontSize: '11px', 
    margin: '0 0 2px 0'
  }
}));

presentClasses.forEach(function(v) {
  var label = classLabels[v];
  var rgb = shelterPalette[v];
  if (label && rgb) {
    var hex = '#' + rgb.map(function(c) {
      var h = c.toString(16);
      return h.length === 1 ? '0' + h : h;
    }).join('');
    
    var colorBox = ui.Label('', {
      backgroundColor: hex,
      padding: '6px', // Reduced from 8px
      margin: '0'
    });
    
    var desc = ui.Label({
      value: label,
      style: {
        margin: '0 0 0 4px', // Reduced spacing
        fontSize: '10px' // Smaller text
      }
    });
    
    var row = ui.Panel({
      widgets: [colorBox, desc],
      layout: ui.Panel.Layout.Flow('horizontal'),
      style: {margin: '1px 0'} // Reduced vertical spacing
    });
    legend.add(row);
  }
});

Map.add(legend);


//////////////////////////////////////////////////////////////////////
// Export layers — ordered to match the layer panel (top to bottom)
exportLayers['Shelter 2020 default wind (10m)'] = defaultWindImg;
exportLayers['Shelter 2020 less wind (10m)'] = lessWindImg;
exportLayers['Shelter 2020 more wind (10m)'] = moreWindImg;
exportLayers['Shelter 2020 default density (10m)'] = defaultDensityImg;
exportLayers['Shelter 2020 less density (10m)'] = lessDensityImg;
exportLayers['Shelter 2020 more density (10m)'] = moreDensityImg;
exportLayers['Tree predictions 2024 (10m)'] = aus2024;
exportLayers['Tree predictions 2020 (10m)'] = aus2020;
exportLayers['Tree predictions 2017 (10m)'] = aus2017;
exportLayers['Canopy Height v2 (1m)'] = chm;
exportLayers['WorldCover 2020 (10m)'] = wc;

//////////////////////////////////////////////////////////////////////
// Export panel
var exportPanel = ui.Panel({
  style: {position: 'bottom-left', padding: '8px', width: '230px'}
});

exportPanel.add(ui.Label('Download layer', {
  fontWeight: 'bold', fontSize: '12px', margin: '0 0 4px 0'
}));

var layerSelect = ui.Select({
  items: Object.keys(exportLayers),
  placeholder: 'Select layer...',
  style: {width: '210px', margin: '0 0 4px 0'}
});
exportPanel.add(layerSelect);

var statusLabel = ui.Label('Draw a rectangle, then click Export.', {
  fontSize: '10px', color: 'gray', margin: '0 0 4px 0', whiteSpace: 'pre-line'
});
exportPanel.add(statusLabel);

var downloadLink = ui.Label('', {
  fontSize: '10px', color: 'blue', textDecoration: 'underline',
  margin: '4px 0 0 0', shown: false
});
exportPanel.add(downloadLink);

var drawButton = ui.Button({
  label: 'Draw region',
  style: {margin: '0 4px 0 0'},
  onClick: function() {
    Map.drawingTools().setShown(true);
    Map.drawingTools().setShape('rectangle');
    Map.drawingTools().clear();
    Map.drawingTools().draw();
    downloadLink.style().set('shown', false);
    statusLabel.setValue('Draw a rectangle on the map.');
  }
});

var MAX_AREA_M2 = 100 * 1e6;  // 10 km × 10 km

var exportButton = ui.Button({
  label: 'Get download link',
  onClick: function() {
    var name = layerSelect.getValue();
    if (!name) {
      statusLabel.setValue('Please select a layer first.');
      return;
    }
    var drawn = Map.drawingTools().layers();
    if (drawn.length() === 0) {
      statusLabel.setValue('Please draw a region first.');
      return;
    }
    var geometry = drawn.get(0).toGeometry();
    statusLabel.setValue('Checking area...');
    downloadLink.style().set('shown', false);
    geometry.area({maxError: 1}).evaluate(function(area) {
      if (area > MAX_AREA_M2) {
        statusLabel.setValue('Region too large (max 10×10 km).\nPlease draw a smaller area.');
        return;
      }
      statusLabel.setValue('Generating link...');
      exportLayers[name].getDownloadURL({
        region: geometry,
        scale: 10,
        format: 'GeoTIFF',
        maxPixels: 1e8
      }, function(url, error) {
        if (error) {
          statusLabel.setValue('Error: try a smaller region.');
        } else {
          statusLabel.setValue('Link ready (expires after download):');
          downloadLink.setValue('Download GeoTIFF');
          downloadLink.setUrl(url);
          downloadLink.style().set('shown', true);
        }
      });
    });
  }
});

Map.drawingTools().onDraw(function() {
  var layers = Map.drawingTools().layers();
  while (layers.length() > 1) {
    layers.remove(layers.get(0));
  }
  Map.drawingTools().stop();
  statusLabel.setValue('Rectangle drawn. Click "Get download link".');
});

exportPanel.add(ui.Panel(
  [drawButton, exportButton],
  ui.Panel.Layout.Flow('horizontal')
));

Map.add(exportPanel);
