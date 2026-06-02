
Map.setOptions('SATELLITE');
Map.drawingTools().setShown(false);


var greenPalette = [
  '#e5f5e0','#c7e9c0','#a1d99b','#74c476','#41ab5d',
  '#238b45','#006d2c','#00441b'
];
var viridisPalette = [
  '#440154','#481567','#482677','#453781','#404788','#39568C',
  '#33638D','#2D708E','#287D8E','#238A8D','#1F968B','#20A387',
  '#29AF7F','#3CBB75','#55C667','#73D055','#95D840','#B8DE29',
  '#DCE319','#FDE725'
];

var brownGreenPalette = [
  '#543005', // dark brown
  '#6f3e08', // lighter brown
  '#995c16',
  '#bc7524',
  '#d3a143', // mid-brown
  '#a6c04a', // light green
  '#78ad51',
  '#4d9950',
  '#238443',
  '#005a2b'  // dark green
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
// Tree confidence
// Map.addLayer(image2, {min: 1, max: 100, palette: brbgPalette}, '2018 Tree confidence (1–100)', false, 0.3);

// // 1m Canopy Height Model
// var chm_masked = image11.updateMask(image11.gt(0));
// Map.addLayer(chm_masked, {min: 1, max: 25, palette: viridisPalette}, 'Milgadara 1m Canopy Height (1–25 m)', false);


///////////////////////////////////////////////////////////
// // Brown to green percent confidence
// Map.addLayer(
//   image2,
//   {min: 1, max: 100, palette: brownGreenPalette},
//   '2018' + ' NSW Tree confidence (1–100%)',
//   false,
//   .65
// );


///////////////////////////////////////////////////////////
// // Tree thresholds
// var threshold=50
// var mask = image2.gt(threshold);
// var imageMasked = image2.updateMask(mask);
// Map.addLayer(imageMasked, {min: threshold, max: 100, palette: ['00FF00']}, '2018 NSW Tree confidence > ' + threshold + '%', false, 0.65);

// var threshold=90
// var mask = image2.gt(threshold);
// var imageMasked = image2.updateMask(mask);
// Map.addLayer(imageMasked, {min: threshold, max: 100, palette: ['00FF00']}, '2018 NSW Tree confidence > ' + threshold + '%', false, 0.65);


///////////////////////////////////////////////////////////
// // Aus trees
var aus2020 = ee.ImageCollection('projects/ee-christopher-bradley/assets/Aus_2020_noxy_predictions').mosaic();
Map.addLayer(
    aus2020.updateMask(aus2020.gt(50)),
    {min: 50, max: 100, palette: ['00FF00']},
    '2020 tree confidence 50%',
    true, 0.65
);
Map.addLayer(
    aus2020.updateMask(aus2020.gt(90)),
    {min: 90, max: 100, palette: ['00FF00']},
    '2020 tree confidence 90%',
    false, 0.65
);

var aus2024 = ee.ImageCollection('projects/ee-christopher-bradley/assets/Aus2024_noxy_predictions').mosaic();
Map.addLayer(
    aus2024.updateMask(aus2024.gt(50)),
    {min: 50, max: 100, palette: ['00FF00']},
    '2024 tree confidence 50%',
    false, 0.65
);
Map.addLayer(
    aus2024.updateMask(aus2024.gt(90)),
    {min: 90, max: 100, palette: ['00FF00']},
    '2024 tree confidence 90%',
    false, 0.65
);

///////////////////////////////////////////////////////////

// Shelter classifications
styleShelterImage(ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-percentmethod_lat10-26'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-percentmethod_lat28-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-percentmethod_lat34-42')
]), '2020 less density method', false);
styleShelterImage(ee.ImageCollection('projects/ee-christopher-bradley/assets/Aus_ag2020_default-percentmethod'), '2020 default density method', false);
styleShelterImage(ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-percentmethod_lat10-26'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-percentmethod_lat28-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-percentmethod_lat34-42')
]), '2020 more density method', false);
styleShelterImage(ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-windmethod_lat10-26'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-windmethod_lat28-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_less-windmethod_lat34-42')
]), '2020 less wind method', false);
styleShelterImage(ee.ImageCollection('projects/ee-christopher-bradley/assets/Aus_ag2020_default-windmethod'), '2020 default wind method', true);
styleShelterImage(ee.ImageCollection([
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-windmethod_lat10-28'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-windmethod_lat28-32'),
  ee.Image('projects/ee-christopher-bradley/assets/Aus_ag2020_more-windmethod_lat34-42')
]), '2020 more wind method', false);


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



// //////////////////////////////////////////////////////////
// // --- Tree Confidence Legend ---
// var legendConfidence = ui.Panel({
//   style: {
//     position: 'bottom-left',
//     padding: '8px 15px'
//   }
// });

// legendConfidence.add(ui.Label({
//   value: 'Tree Confidence (1–100)',
//   style: {
//     fontWeight: 'bold',
//     fontSize: '14px',
//     margin: '0 0 4px 0'
//   }
// }));

// // Add color bar gradient
// var colorBar = ui.Thumbnail({
//   image: ee.Image.pixelLonLat().select(0)
//     .multiply(0)
//     .add(ee.Image.pixelLonLat().select(0).unitScale(0, 1)),
//   params: {
//     bbox: [0, 0, 1, 0.1],
//     dimensions: '150x10',
//     format: 'png',
//     min: 0,
//     max: 1,
//     palette: brownGreenPalette
//   },
//   style: {stretch: 'horizontal', margin: '0 0 6px 0'}
// });
// legendConfidence.add(colorBar);

// Map.add(legendConfidence);


