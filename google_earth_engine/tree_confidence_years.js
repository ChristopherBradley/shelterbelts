// Tree confidence year series (2017-2025) over the ag extent, for visual QA before a national run.
// Each year is an ImageCollection of 148 region tifs (0-100 = % tree confidence, 255 = nodata),
// uploaded by pbs_scripts/upload_predictions_years.pbs.
// All 9 years are added as map layers up front; the slider just changes which one is shown, so
// flipping back to a year you've already viewed doesn't need to re-render tiles.

Map.setOptions('SATELLITE');
Map.setCenter(148.471268, -34.389131, 12);  // (lon, lat, zoom)
Map.drawingTools().setShown(false);
Map.drawingTools().setDrawModes(['rectangle']);

// Use a single dedicated layer for the export region so only one rectangle can ever exist at a time.
while (Map.drawingTools().layers().length() > 0) {
  Map.drawingTools().layers().remove(Map.drawingTools().layers().get(0));
}
var exportRegionLayer = ui.Map.GeometryLayer({
  geometries: null, name: 'exportRegion', color: 'yellow'
});
Map.drawingTools().layers().add(exportRegionLayer);

var years = [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025];
var assetRoot = 'projects/christopher-bradley-phd/assets/';

var palette = ['#ffffcc', '#d9f0a3', '#addd8e', '#78c679', '#41ab5d', '#238443', '#005a32'];
var vis = {min: 0, max: 100, palette: palette};

var state = {year: 2025, minConfidence: 0, opacity: 1};

var images = {};
years.forEach(function(y) {
  images[y] = ee.ImageCollection(assetRoot + 'Aus' + y + '_ag_predictions').mosaic();
});

function styled(y) {
  var img = images[y];
  return img.updateMask(img.gte(state.minConfidence));
}

var layers = {};
years.forEach(function(y) {
  var layer = ui.Map.Layer(styled(y), vis, 'Tree confidence ' + y, y === state.year, state.opacity);
  layers[y] = layer;
  Map.layers().add(layer);
});

function showYear(y) {
  state.year = y;
  years.forEach(function(k) { layers[k].setShown(k === y); });
  yearLabel.setValue('Showing ' + y);
}


///////////////////////////////////////////////////////////////////////////
// Control panel

var panel = ui.Panel({style: {position: 'top-left', padding: '8px', width: '260px'}});
panel.add(ui.Label('Tree confidence', {fontWeight: 'bold', fontSize: '16px', margin: '0 0 6px 0'}));

var yearLabel = ui.Label('Showing ' + state.year, {fontWeight: 'bold'});
panel.add(yearLabel);

var yearSlider = ui.Slider({
  min: years[0], max: years[years.length - 1], step: 1, value: state.year,
  style: {stretch: 'horizontal'},
  onChange: showYear
});
panel.add(yearSlider);

function step(delta) {
  var i = years.indexOf(state.year) + delta;
  if (i >= 0 && i < years.length) { yearSlider.setValue(years[i]); }  // triggers showYear
}
panel.add(ui.Panel([
  ui.Button({label: '◀ Previous', onClick: function() { step(-1); }}),
  ui.Button({label: 'Next ▶', onClick: function() { step(1); }})
], ui.Panel.Layout.flow('horizontal')));

panel.add(ui.Label('Hide confidence below (%)', {margin: '10px 8px 0 8px'}));
panel.add(ui.Slider({
  min: 0, max: 100, step: 5, value: state.minConfidence, style: {stretch: 'horizontal'},
  onChange: function(v) {
    state.minConfidence = v;
    years.forEach(function(y) { layers[y].setEeObject(styled(y)); });
  }
}));

panel.add(ui.Label('Opacity', {margin: '10px 8px 0 8px'}));
panel.add(ui.Slider({
  min: 0, max: 1, step: 0.05, value: state.opacity, style: {stretch: 'horizontal'},
  onChange: function(v) {
    state.opacity = v;
    years.forEach(function(y) { layers[y].setOpacity(v); });
  }
}));

// Colour bar legend
var colourBar = ui.Thumbnail({
  image: ee.Image.pixelLonLat().select(0),
  params: {bbox: [0, 0, 1, 0.1], dimensions: '200x12', format: 'png', min: 0, max: 1, palette: palette},
  style: {stretch: 'horizontal', margin: '10px 8px 0 8px', maxHeight: '20px'}
});
panel.add(colourBar);
panel.add(ui.Panel([
  ui.Label('0%', {margin: '2px 8px'}),
  ui.Label('50%', {margin: '2px 8px', textAlign: 'center', stretch: 'horizontal'}),
  ui.Label('100%', {margin: '2px 8px'})
], ui.Panel.Layout.flow('horizontal')));

Map.add(panel);


//////////////////////////////////////////////////////////////////////
// Export panel
// Downloads the raw confidence values (0-100, 255 = nodata), not the coloured map, and ignores the
// "hide confidence below" slider, so the tif can be thresholded however you like afterwards.
var ALL_YEARS = 'All 9 years (9-band stack)';
var exportLayers = {};
years.forEach(function(y) {
  exportLayers['Tree confidence ' + y] = images[y].unmask(255).rename('confidence_' + y);
});
exportLayers[ALL_YEARS] = ee.Image.cat(years.map(function(y) {
  return exportLayers['Tree confidence ' + y];
}));

var exportPanel = ui.Panel({
  style: {position: 'bottom-left', padding: '8px', width: '230px'}
});

exportPanel.add(ui.Label('Download a layer', {
  fontWeight: 'bold', fontSize: '13px', margin: '0 0 6px 0'
}));

// Widgets are defined here but added to the panel further down, in stepped order.
var layerSelect = ui.Select({
  items: Object.keys(exportLayers),
  placeholder: 'Select layer...',
  style: {width: '210px', margin: '0 0 2px 0'},
  onChange: function() {
    statusLabel.setValue('Layer chosen. Now click "Get download link"');
  }
});

var statusLabel = ui.Label('Click "Draw Region" above', {
  fontSize: '10px', color: 'gray', margin: '8px 0 0 0', whiteSpace: 'pre-line'
});

var downloadLink = ui.Label('', {
  fontSize: '11px', color: 'blue', textDecoration: 'underline',
  margin: '2px 0 0 0', shown: false
});

var drawButton = ui.Button({
  label: 'Draw region',
  style: {stretch: 'horizontal', margin: '0'},
  onClick: function() {
    Map.drawingTools().setShown(false);
    Map.drawingTools().setShape('rectangle');

    // clear() doesn't empty the custom layer, so remove its geometries directly
    var geoms = exportRegionLayer.geometries();
    while (geoms.length() > 0) {
      geoms.remove(geoms.get(0));
    }
    Map.drawingTools().draw();
    downloadLink.style().set('shown', false);
    statusLabel.setValue('Now draw a rectangle on the map');
  }
});

var MAX_AREA_M2 = 100 * 1e6;  // 10 km × 10 km

var exportButton = ui.Button({
  label: 'Get download link',
  style: {stretch: 'horizontal', margin: '0'},
  onClick: function() {
    var name = layerSelect.getValue();
    if (!name) {
      statusLabel.setValue('Please select a layer first.');
      return;
    }
    var geoms = exportRegionLayer.geometries();
    if (geoms.length() === 0) {
      statusLabel.setValue('Please draw a region first.');
      return;
    }
    // Safety net: if more than one rectangle was drawn, keep only the newest so
    // the download is always a single region.
    while (geoms.length() > 1) {
      geoms.remove(geoms.get(0));
    }
    var geometry = exportRegionLayer.toGeometry();
    statusLabel.setValue('Checking area...');
    downloadLink.style().set('shown', false);
    geometry.area({maxError: 1}).evaluate(function(area) {
      if (area > MAX_AREA_M2) {
        statusLabel.setValue('Region too large (max 10×10 km).\nPlease draw a smaller area.');
        return;
      }
      statusLabel.setValue('Generating link...');
      exportLayers[name].toByte().getDownloadURL({
        name: name === ALL_YEARS ? 'tree_confidence_2017_2025' : name.replace(/ /g, '_').toLowerCase(),
        region: geometry,
        crs: 'EPSG:3857',  // the source tifs' native 10 m grid, so no reprojection
        scale: 10,
        format: 'GeoTIFF',
        maxPixels: 1e8
      }, function(url, error) {
        if (error) {
          statusLabel.setValue('Error: try a smaller region.');
        } else {
          statusLabel.setValue('Link ready (expires after download):');
          downloadLink.setValue('Download GeoTIFF (0-100, 255 = nodata)');
          downloadLink.setUrl(url);
          downloadLink.style().set('shown', true);
        }
      });
    });
  }
});

// Whenever a draw event occurs, prune to just newest rectangle.
Map.drawingTools().onDraw(function() {
  var geoms = exportRegionLayer.geometries();
  while (geoms.length() > 1) {
    geoms.remove(geoms.get(0));
  }
  if (layerSelect.getValue()) {
    statusLabel.setValue('Layer selected. Now click "Get download link"');
  } else {
    statusLabel.setValue('Rectangle drawn. Now click "Select Layer..."');
  }
});

// Assemble the panel as three numbered steps.
var stepStyle = {fontSize: '11px', fontWeight: 'bold', margin: '6px 0 3px 0'};

exportPanel.add(ui.Label('Step 1: Draw a rectangle on the map', stepStyle));
exportPanel.add(drawButton);

exportPanel.add(ui.Label('Step 2: Choose the layer to download', stepStyle));
exportPanel.add(layerSelect);

exportPanel.add(ui.Label('Step 3: Get the download link, then click it', stepStyle));
exportPanel.add(exportButton);

exportPanel.add(statusLabel);
exportPanel.add(downloadLink);

Map.add(exportPanel);
