// ============================================================================
// GEDI Point Extraction for Building Height Proxies
// Filters: 
// 1. Geographic Area (geometry)
// 2. Quality Flag = 1 (Valid Waveform)
// 3. Sensitivity > 0.9 (High Energy/Ground Detection Probability)
// ============================================================================

// 1. ROI SETUP
// ----------------------------------------------------------------------------
// If no geometry is drawn, defaults to a test area (London).
if (typeof geometry === 'undefined') {
  var geometry = ee.Geometry.Polygon([
    [-0.15, 51.48], [-0.05, 51.48], [-0.05, 51.52], [-0.15, 51.52]
  ]);
  print('WARNING: No geometry defined. Using test area (London).');
}
Map.centerObject(geometry, 12);
Map.addLayer(geometry, {color: 'red'}, 'Region of Interest');

// 2. LOAD DATASETS (GEDI Monthly Raster)
// ----------------------------------------------------------------------------
// Using Monthly Raster to aggregate data and avoid folder errors.
var gediMonthly = ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY")
  .filterDate('2019-01-01', '2024-12-31')
  .filterBounds(geometry);

// 3. PROCESSING & FILTERING
// ----------------------------------------------------------------------------
// We create a mask that satisfies BOTH Quality and Sensitivity requirements.
var gediFiltered = gediMonthly.map(function(img) {
  var quality = img.select('quality_flag');
  var sensitivity = img.select('sensitivity');
  
  // The Strict Filter: 
  // Quality must be 1 AND Sensitivity must be >= 0.9
  var combinedMask = quality.eq(1).and(sensitivity.gte(0.9));
  
  return img.updateMask(combinedMask);
});

// 4. AGGREGATE & CONVERT TO POINTS
// ----------------------------------------------------------------------------
// Collapse time series to single layer (Mean values for height metrics)
var gediComposite = gediFiltered
  .select(['rh85', 'rh95'])
  .mean()
  .clip(geometry);

// Convert pixels to Vector Points (Centroids)
// Scale: 25m (Native GEDI footprint size)
var gediPoints = gediComposite.sample({
  region: geometry,
  scale: 25, 
  geometries: true // Creates the Point Geometry
});

// 5. ADD COORDINATES (EPSG:3857)
// ----------------------------------------------------------------------------
// Add Web Mercator (Meters) coordinates as attributes
var exportCollection = gediPoints.map(function(feature) {
  // Transform geometry to EPSG:3857
  var geom3857 = feature.geometry().transform('EPSG:3857');
  var coords = geom3857.coordinates();
  
  return feature.set({
    'x_3857': coords.get(0),
    'y_3857': coords.get(1)
  });
});

print('Final Point Count:', exportCollection.size());
Map.addLayer(exportCollection, {color: '00FF00'}, 'High Quality GEDI Points');

// 6. EXPORT
// ----------------------------------------------------------------------------
Export.table.toDrive({
  collection: exportCollection,
  description: 'GEDI_Points_',
  fileFormat: 'GeoJSON',
  // Exporting Height Metrics + Coordinates
  selectors: ['rh85', 'rh95', 'x_3857', 'y_3857', '.geo'] 
});