
Map.setOptions('SATELLITE');
Map.drawingTools().setShown(false);

var viridis = ['440154','482878','3e4989','31688e','26828e','1f9e89','35b779','6ece58','b5de2b','fde725'];
var HEIGHT_MIN = 0;
var HEIGHT_MAX = 25;

///////////////////////////////////////////////////////////
// LiDAR CHM (background)
var chm = ee.Image('projects/christopher-bradley-phd/assets/1m_chm_7x7');
Map.addLayer(
  chm.updateMask(chm.gt(0)),
  {min: HEIGHT_MIN, max: HEIGHT_MAX, palette: viridis},
  'LiDAR CHM 1m', true, 1.0
);

Map.centerObject(chm, 17);


///////////////////////////////////////////////////////////
// Tree crown polygons (foreground, coloured by height)
var crowns = ee.FeatureCollection('projects/christopher-bradley-phd/assets/1m_crowns_7x7');

// // Fill polygons by height — adjust property name if needed (e.g. 'max_height', 'mean_height')
// var crownsFilled = ee.Image().float().paint({
//   featureCollection: crowns,
//   color: 'p95_height'
// });
// Map.addLayer(
//   crownsFilled.updateMask(crownsFilled.gt(0)),
//   {min: HEIGHT_MIN, max: HEIGHT_MAX, palette: viridis},
//   'Tree crowns (filled by height)', true, 0.8
// );

// White outline so individual crowns are distinguishable
var crownsOutline = ee.Image().paint({
  featureCollection: crowns,
  color: 1,
  width: 1
});
Map.addLayer(crownsOutline, {palette: ['ffffff']}, 'Tree crown outlines', true, 0.6);

///////////////////////////////////////////////////////////
// Colorbar legend
var legend = ui.Panel({
  style: {position: 'bottom-right', padding: '8px', width: '200px'}
});

legend.add(ui.Label('Canopy height (m)', {fontWeight: 'bold', fontSize: '12px', margin: '0 0 6px 0'}));

var gradient = ui.Panel({layout: ui.Panel.Layout.Flow('horizontal'), style: {margin: '0 0 2px 0'}});
viridis.forEach(function(hex) {
  gradient.add(ui.Label('', {
    backgroundColor: '#' + hex,
    padding: '8px',
    margin: '0',
    width: '16px'
  }));
});
legend.add(gradient);

var tickRow = ui.Panel({layout: ui.Panel.Layout.Flow('horizontal')});
tickRow.add(ui.Label(String(HEIGHT_MIN), {fontSize: '10px', margin: '0'}));
tickRow.add(ui.Label('', {stretch: 'horizontal'}));
tickRow.add(ui.Label(String(HEIGHT_MAX) + 'm', {fontSize: '10px', margin: '0'}));
legend.add(tickRow);

Map.add(legend);
