
Map.setOptions('SATELLITE');
Map.drawingTools().setShown(false);  // Hide the point/line/polygon/rectangle draw icons
Map.setControlVisibility({layerList: false});  // Hide the native Layers panel — replaced with a smaller custom one below

function lastAddedLayer() {
  return Map.layers().get(Map.layers().length() - 1);
}

///////////////////////////////////////////////////////////////////////////
// Setup for colour schemes

function blendRgb(rgbA, rgbB, t) {
  // t=0 -> rgbA, t=1 -> rgbB
  return [0, 1, 2].map(function(k) {
    return Math.round(rgbA[k] + (rgbB[k] - rgbA[k]) * t);
  });
}

function mergeDicts() {
  var out = {};
  for (var i = 0; i < arguments.length; i++) {
    var d = arguments[i];
    Object.keys(d).forEach(function(k) { out[k] = d[k]; });
  }
  return out;
}

// Base categories
var basePalette = {
  0:   [255, 255, 255],  // White: Outside region / not trees
  10:  [0, 100, 0],      // Green: Worldcover tree cover
  11:  [122, 82, 0],     // Brown: Scattered Trees
  12:  [8, 79, 0],       // Dark green: Patch core
  13:  [14, 138, 0],     // Medium green: Patch edge
  14:  [22, 212, 0],     // Bright green: Other trees
  15:  [29, 153, 105],   // Bluey green: Trees in gullies
  16:  [127, 168, 57],   // Olive: Trees on ridges
  17:  [129, 146, 124],  // Silver: Trees next to roads
  18:  [190, 160, 60],   // Light brown: Linear patches (shelterbelts)
  19:  [165, 195, 45],   // Bright olive green: Non-linear patches
  20:  [255, 187, 34],   // Orange: Shrubs
  30:  [255, 255, 76],   // Yellow: Worldcover grassland (unsheltered)
  40:  [240, 150, 255],  // Pink: Worldcover cropland (unsheltered)
  50:  [250, 0, 0],      // Red: Built-up
  60:  [180, 180, 180],  // Grey: Bare
  70:  [240, 240, 240],  // White: Snow
  80:  [0, 100, 200],    // Blue: Water
  90:  [0, 150, 160],    // Worldcover wetland
  95:  [0, 207, 117],    // Worldcover mangroves
  100: [250, 230, 160],  // Worldcover moss and lichen
};

// Values used to calculate the sheltered colours
var treeSourceColourByDigit = {
  2: basePalette[12],  // Patch core
  3: basePalette[13],  // Patch edge
  4: basePalette[14],  // Other trees
  5: basePalette[15],  // Trees in gullies
  6: basePalette[16],  // Trees on ridges
  7: basePalette[17],  // Trees next to roads
  8: basePalette[18],  // Linear patches
  9: basePalette[19],  // Non-linear patches
};
var treeSourceLabelByDigit = {
  2: 'Patch Core',
  3: 'Patch Edge',
  4: 'Other Trees',
  5: 'Trees in Gullies',
  6: 'Trees on Ridges',
  7: 'Trees next to Roads',
  8: 'Linear Patches',
  9: 'Non-linear Patches',
};
var shelterDigits = [2, 3, 4, 5, 6, 7, 8, 9];

// Blend the (grassland) base colour towards the actual sheltering tree's colour
var treeBlendAmount = 0.6;  // 0 = pure farmland colour, 1 = pure tree colour
var shelteredByDigit = {};
shelterDigits.forEach(function(digit) {
  shelteredByDigit[digit] = blendRgb(basePalette[30], treeSourceColourByDigit[digit], treeBlendAmount);
});

var grasslandByTreeType = {30: basePalette[30], 31: [0, 0, 0]};
var croplandByTreeType  = {40: basePalette[30], 41: [0, 0, 0]};
shelterDigits.forEach(function(digit) {
  grasslandByTreeType[30 + digit] = shelteredByDigit[digit];
  croplandByTreeType[40 + digit]  = shelteredByDigit[digit];
});

var shelterPalette = mergeDicts(basePalette, grasslandByTreeType, croplandByTreeType);


///////////////////////////////////////////////////////////////////////////
// Shared styling helper for both shelter categories & opportunities
var fullyTransparentClasses = [0, 30, 31, 40, 41];
var partialTransparentClasses = [32, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47, 48, 49];
var shelteredTransparency = 0.35;

function styleCategoricalImage(img, layerName, checkbox, palette, transparency) {
  // Normalize ImageCollection to Image
  if (img instanceof ee.ImageCollection) {
    img = img.mosaic();
  }

  // Remove fully transparent classes
  var classKeys = Object.keys(palette)
    .map(function(k){ return parseInt(k, 10); })
    .filter(function(k){ return fullyTransparentClasses.indexOf(k) === -1; })
    .sort(function(a,b){ return a - b; });

  // Create a consecutive colour scheme
  var paletteHex = classKeys.map(function(k){
    var rgb = palette[k];
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

  // Add partial transparency for sheltered farmland, so the shelter attribution shows through gently
  var mask = baseMask;
  partialTransparentClasses.forEach(function(cls) {
    mask = mask.where(classesImg.eq(cls), shelteredTransparency);
  });
  var styled = remapped.updateMask(mask);

  Map.addLayer(styled,
               {min: 0, max: targetIndices.length - 1, palette: paletteHex},
               layerName, checkbox, transparency);

  return styled;
}

Map.setCenter(148.471268, -34.389131, 12);  // (lon, lat, zoom)


///////////////////////////////////////////////////////////
// Parameter combinations — every combo has a shelter categories asset collection, but only the
// default combo also has shelter distances + planting opportunities (matches the main app's
// trimmed 6-layer set instead of the full 4 x 3 grid).
var paramCombos = [
  {key: 'default_windmethod',    label: 'default'},
  {key: 'less_windmethod',       label: 'low'},
  {key: 'more_windmethod',       label: 'high'},
  {key: 'default_percentmethod', label: 'percent'},
];
var defaultComboKey = 'default_windmethod';
var assetVersionTag = 'v3';
// Opportunities alone is bumped to v4 — see the matching note in shelterbelts_aus_gch.js.
var opportunitiesVersionTag = 'v4';

var distanceMax = 20;
var ylGn = ['ffffe5', 'f7fcb9', 'd9f0a3', 'addd8e', '78c679', '41ab5d', '238443', '006837', '004529'];
var ylGnInvertedForDistance = ylGn.slice().reverse();

// NOTE: the GEE Layers panel shows the most-recently-added layer at the TOP,
// i.e. panel order is the REVERSE of the order Map.addLayer is called in.
// So everything below is added in reverse of the desired panel order
// (bottom-most layer first, top-most layer last).

///////////////////////////////////////////////////////////
// Canopy Height Model v2 (Meta & WRI) — added first so it ends up at the bottom
var viridis = ['440154','482878','3e4989','31688e','26828e','1f9e89','35b779','6ece58','b5de2b','fde725'];
var chm = ee.ImageCollection('projects/meta-forest-monitoring-okw37/assets/CanopyHeight').mosaic();
Map.addLayer(
  chm.updateMask(chm.gt(0)),
  {min: 0, max: 25, palette: viridis},
  'Meta Canopy Height v2', false, 0.8
);
var chmLayerRef = lastAddedLayer();

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
var wcLayerRef = lastAddedLayer();

var opportunitiesRefByCombo = {};
var shelterDistancesRefByCombo = {};
var shelterCategoriesRefByCombo = {};

// Planting opportunities — default combo only (matches shelterbelts_aus_gch.js)
(function() {
  var combo = paramCombos.filter(function(c) { return c.key === defaultComboKey; })[0];
  var assetPrefix = 'projects/ee-christopher-bradley/assets/Aus2025_ag_gch_' + combo.key + '_' + opportunitiesVersionTag + '_';
  var opportunitiesImg = ee.ImageCollection(assetPrefix + 'opportunities').mosaic();
  styleCategoricalImage(
    opportunitiesImg, 'Opportunities (' + combo.label + ')', false, shelterPalette, 1);
  opportunitiesRefByCombo[combo.key] = lastAddedLayer();
})();

// Shelter distances — default combo only (matches shelterbelts_aus_gch.js)
(function() {
  var combo = paramCombos.filter(function(c) { return c.key === defaultComboKey; })[0];
  var assetPrefix = 'projects/ee-christopher-bradley/assets/Aus2025_ag_gch_' + combo.key + '_' + assetVersionTag + '_';
  var shelterDistancesImg = ee.ImageCollection(assetPrefix + 'shelter_distances').mosaic();
  var shelterDistances = shelterDistancesImg.updateMask(shelterDistancesImg.gt(0));
  Map.addLayer(
    shelterDistances,
    {min: 1, max: distanceMax, palette: ylGnInvertedForDistance},
    'Distances (' + combo.label + ')', false, 1
  );
  shelterDistancesRefByCombo[combo.key] = lastAddedLayer();
})();

// Shelter categories (added in reverse combo order, last one added ends up on top) — every combo
paramCombos.slice().reverse().forEach(function(combo) {
  var isDefault = combo.key === defaultComboKey;
  var assetPrefix = 'projects/ee-christopher-bradley/assets/Aus2025_ag_gch_' + combo.key + '_' + assetVersionTag + '_';
  var shelterCategoriesImg = ee.ImageCollection(assetPrefix + 'shelter_categories').mosaic();
  styleCategoricalImage(
    shelterCategoriesImg, 'Categories (' + combo.label + ')', isDefault, shelterPalette, 1);
  shelterCategoriesRefByCombo[combo.key] = lastAddedLayer();
});


//////////////////////////////////////////////////////////
// Create an info panel
var infoPanel = ui.Panel({
  style: {
    width: '120px',
    position: 'top-right',
    padding: '2px',
    backgroundColor: 'white'
  }
});

// Title
infoPanel.add(ui.Label({
  value: 'Shelterbelts 2025',
  style: {fontWeight: 'bold', fontSize: '8px', margin: '0 0 1px 0'}
}));

infoPanel.add(ui.Label(
  'Work-in-progress. Code and details here:'
    , {whiteSpace: 'pre-line', fontSize: '6px', margin: '0 0 0px 0'}
));

infoPanel.add(ui.Label(
  'https://github.com/ChristopherBradley/shelterbelts',
  {fontSize: '6px', color: 'blue', textDecoration: 'underline', margin: '0 0 2px 0'},
  'https://github.com/ChristopherBradley/shelterbelts'
));
infoPanel.add(ui.Label(
  'Feedback: \nchristopher.bradley@anu.edu.au\n'
    , {whiteSpace: 'pre-line', fontSize: '6px', margin: '0 0 0px 0'}
));

// Add the panel to the map
Map.add(infoPanel);


//////////////////////////////////////////////////////////////////////
// Custom Layers panel (~60% the size of the native one, which we hid above).
// Added to the map after the info panel and before the legend, so it
// stacks between them in the top-right corner.
var layersPanel = ui.Panel({
  style: {
    position: 'top-right',
    padding: '2px 3px',
    maxWidth: '95px',
    maxHeight: '35vh',
    backgroundColor: 'white'
  }
});

layersPanel.add(ui.Label({
  value: 'Layers',
  style: {fontWeight: 'bold', fontSize: '7px', margin: '0 0 1px 0'}
}));

function addLayerToggle(label, layerRef, checked) {
  layersPanel.add(ui.Checkbox({
    label: label,
    value: checked,
    style: {fontSize: '6px', margin: '1px 0'},
    onChange: function(shown) {
      layerRef.setShown(shown);
    }
  }));
}

paramCombos.forEach(function(combo) {
  addLayerToggle('Categories (' + combo.label + ')', shelterCategoriesRefByCombo[combo.key], combo.key === defaultComboKey);
});
(function() {
  var combo = paramCombos.filter(function(c) { return c.key === defaultComboKey; })[0];
  addLayerToggle('Distances (' + combo.label + ')', shelterDistancesRefByCombo[combo.key], false);
  addLayerToggle('Opportunities (' + combo.label + ')', opportunitiesRefByCombo[combo.key], false);
})();
addLayerToggle('WorldCover 2020', wcLayerRef, false);
addLayerToggle('Canopy Height', chmLayerRef, false);

Map.add(layersPanel);


//////////////////////////////////////////////////////////////////////
// Adding a legend.
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
  30: 'Grassland (unsheltered)',
  40: 'Cropland (unsheltered)',
  50: 'Built-up',
  60: 'Bare',
  70: 'Snow and ice',
  80: 'Permanent water bodies',
  90: 'Herbaceous wetland',
  95: 'Mangroves',
  100: 'Moss and lichen'
};
shelterDigits.forEach(function(digit) {
  classLabels[30 + digit] = 'Sheltered by ' + treeSourceLabelByDigit[digit];
});

// Manually choosing values, because there are some that don't show up enough to be worthwhile,
var presentClasses = [11, 12, 13, 15, 17, 18, 19, 20,
                       50, 60, 80,
                      33, 35, 37, 38, 39];

var legend = ui.Panel({
  style: {
    position: 'top-right',
    padding: '2px 4px',
    maxWidth: '112px',
    maxHeight: '22vh'
  }
});

legend.add(ui.Label({
  value: 'Shelter Categories',
  style: {
    fontWeight: 'bold',
    fontSize: '7px',
    margin: '0 0 1px 0'
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
      padding: '3px',
      margin: '0'
    });

    var desc = ui.Label({
      value: label,
      style: {
        margin: '0 0 0 2px',
        fontSize: '6px'
      }
    });

    var row = ui.Panel({
      widgets: [colorBox, desc],
      layout: ui.Panel.Layout.Flow('horizontal'),
      style: {margin: '1px 0'}
    });
    legend.add(row);
  }
});

Map.add(legend);
