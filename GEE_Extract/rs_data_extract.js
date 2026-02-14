// === 1. SETUP & ROI ===
var roi = typeof geometry !== 'undefined' ? geometry : ee.Geometry.Polygon([
  [[114.0, 22.15], [114.3, 22.15], [114.3, 22.55], [114.0, 22.55]] 
]);
Map.addLayer(roi, {color: 'red'}, 'Region of Interest');

// === 2. DATA LOADING & PRE-PROCESSING ===

// --- Sentinel-1 (SAR) ---
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filterDate('2019-01-01', '2022-12-31')
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .map(function(image) {
    var vvh = image.select('VV').multiply(ee.Image(5).pow(image.select('VH'))).rename('VVH');
    return image.select(['VV', 'VH']).addBands(vvh).toFloat();
  });

// --- Sentinel-2 (Optical) ---
function maskS2clouds(image) {
  var scl = image.select('SCL');
  var validPixels = scl.eq(4).or(scl.eq(5)).or(scl.eq(6)).or(scl.eq(7)).or(scl.eq(11));
  return image.updateMask(validPixels).divide(10000);
}

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2019-01-01', '2022-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
  .map(maskS2clouds)
  .map(function(image) {
    var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
    var lswi = image.normalizedDifference(['B8', 'B11']).rename('LSWI');
    var mndwi = image.normalizedDifference(['B3', 'B11']).rename('mNDWI');
    var ndbi = image.normalizedDifference(['B11', 'B8']).rename('NDBI');
    
    var evi = image.expression(
      '2.5 * ((B8 - B4) / (B8 + 6 * B4 + 7.5 * B2 + 1))', 
      {'B8': image.select('B8'), 'B4': image.select('B4'), 'B2': image.select('B2')}
    ).rename('EVI');

    var csi = image.expression(
      '((B2 + B4) - (B3 + B8 + B11)) / ((B2 + B4) + (B3 + B8 + B11))', 
      {'B2': image.select('B2'), 'B3': image.select('B3'), 'B4': image.select('B4'), 'B8': image.select('B8'), 'B11': image.select('B11')}
    ).rename('CSI');

    return image.select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12'])
                .addBands([ndvi, evi, lswi, mndwi, ndbi, csi])
                .toFloat();
  });

// --- Aggregators (Manual Calculation) ---
function getTemporalStats(collection) {
  var mean = collection.mean();
  var max = collection.max();
  
  // Manual Standard Deviation
  var squared = collection.map(function(img) { return img.pow(2); });
  var mean_of_squared = squared.mean();
  var mean_squared = mean.pow(2);
  
  // Explicit String casting
  var stdDev = mean_of_squared.subtract(mean_squared).sqrt().rename(
    mean.bandNames().map(function(n) { return ee.String(n).cat('_stdDev'); })
  );
  
  mean = mean.rename(mean.bandNames().map(function(n) { return ee.String(n).cat('_mean'); }));
  max = max.rename(max.bandNames().map(function(n) { return ee.String(n).cat('_max'); }));

  return mean.addBands(max).addBands(stdDev).clip(roi).toFloat();
}

var s1_stats = getTemporalStats(s1);
var s2_stats = getTemporalStats(s2);

var s2_indices_stats = s2_stats.select(['NDVI.*', 'EVI.*', 'LSWI.*', 'mNDWI.*', 'NDBI.*', 'CSI.*']);

// === 3. ANCILLARY DATA ===
var srtm = ee.Image('CGIAR/SRTM90_V4').clip(roi).rename('SRTM_DEM');
var copernicus = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').mosaic().clip(roi).rename('Copernicus_DEM');
var alos = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM').mosaic().clip(roi).rename('ALOS_DEM');
var wsf = ee.Image('DLR/WSF/WSF2015/v1').clip(roi).rename('WSF2015');

// Manual Slope Calculation (No libraries)
var grad = srtm.gradient();
var dx = grad.select('x');
var dy = grad.select('y');
var slope = dx.pow(2).add(dy.pow(2)).sqrt().atan().multiply(180).divide(3.14159).rename('Slope');

var viirs = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')
  .filterDate('2021-01-01', '2021-12-31')
  .select('avg_rad')
  .mean()
  .clip(roi)
  .rename('VIIRS_Nighttime');

var population = ee.ImageCollection("WorldPop/GP/100m/pop_age_sex")
  .filterDate('2020-01-01', '2020-12-31')
  .select('population')
  .mean()
  .clip(roi)
  .rename('Population');

var latLon = ee.Image.pixelLonLat().clip(roi);

var diff_Cop_SRTM = copernicus.subtract(srtm).rename('Diff_Cop_SRTM');
var diff_Cop_ALOS = copernicus.subtract(alos).rename('Diff_Cop_ALOS');
var diff_ALOS_SRTM = alos.subtract(srtm).rename('Diff_ALOS_SRTM');

// === 4. SPATIAL NEIGHBORHOOD STATS ===
function addNeighborhoodStats(image, radius, radiusName) {
  var baseBands = image.select('.*_mean');
  
  // Explicit String casting
  var spatialMean = baseBands.focal_mean(radius, 'circle', 'meters')
    .rename(baseBands.bandNames().map(function(b){ return ee.String(b).cat('_spatial_' + radiusName + '_mean') }));

  var spatialMax = baseBands.focal_max(radius, 'circle', 'meters')
    .rename(baseBands.bandNames().map(function(b){ return ee.String(b).cat('_spatial_' + radiusName + '_max') }));

  return spatialMean.addBands(spatialMax).toFloat();
}

// === 5. EXPORTS ===
// Replaced Object.assign with explicit definitions for compatibility
var radii = [50, 100, 150, 200];
radii.forEach(function(r) {
  var name = r + 'm';
  
  Export.image.toDrive({
      image: addNeighborhoodStats(s1_stats, r, name),
      description: 'S1_Spatial_' + name,
      scale: 10,
      region: roi,
      maxPixels: 1e13
  });

  Export.image.toDrive({
      image: addNeighborhoodStats(s2_indices_stats, r, name),
      description: 'S2_Indices_Spatial_' + name,
      scale: 10,
      region: roi,
      maxPixels: 1e13
  });
});

Export.image.toDrive({
  image: s2_indices_stats, 
  description: 'S2_Indices_Temporal_Stats', 
  scale: 10,
  region: roi,
  maxPixels: 1e13
});

Export.image.toDrive({
  image: s1_stats, 
  description: 'S1_Temporal_Stats', 
  scale: 10,
  region: roi,
  maxPixels: 1e13
});

var ancillary = wsf
  .addBands(copernicus).addBands(srtm).addBands(alos)
  .addBands(slope).addBands(viirs)
  .addBands(latLon).addBands(population)
  .addBands(diff_Cop_SRTM).addBands(diff_Cop_ALOS).addBands(diff_ALOS_SRTM)
  .toFloat();

Export.image.toDrive({
  image: ancillary, 
  description: 'Ancillary_Data_Stack',
  scale: 30, // 30m for ancillary
  region: roi,
  maxPixels: 1e13
});

var gedi = ee.FeatureCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY')
  .filterBounds(roi)
  .select(['elevation', 'height_lastbin', 'height_bin0']);

Export.table.toDrive({
  collection: gedi,
  description: 'GEDI_Points',
  fileFormat: 'CSV'
});

// JRC GHSL Data
var ghs_built_2018 = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_H").filterDate('2018-01-01', '2019-01-01').first();
var ghs_surface_2020 = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S").filterDate('2020-01-01', '2021-01-01').first();
var ghs_vol_2020 = ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_V").filterDate('2020-01-01', '2021-01-01').first();

if (ghs_built_2018) {
  Export.image.toDrive({
    image: ghs_built_2018.clip(roi), 
    description: 'GHSL_Height_2018',
    scale: 30,
    region: roi,
    maxPixels: 1e13
  });
}

if (ghs_surface_2020) {
  Export.image.toDrive({
    image: ghs_surface_2020.clip(roi), 
    description: 'GHSL_Surface_2020',
    scale: 30,
    region: roi,
    maxPixels: 1e13
  });
}

if (ghs_vol_2020) {
  Export.image.toDrive({
    image: ghs_vol_2020.clip(roi), 
    description: 'GHSL_Volume_2020',
    scale: 30,
    region: roi,
    maxPixels: 1e13
  });
}

Map.addLayer(s2_indices_stats.select('NDVI_mean'), {min: 0, max: 0.8, palette: ['red', 'yellow', 'green']}, 'NDVI Mean');
