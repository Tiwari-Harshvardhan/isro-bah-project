// 1. Load the GADM country boundaries and filter for Madhya Pradesh districts
var districts = ee.FeatureCollection("FAO/GAUL/2015/level2");
var mpDistricts = districts.filter(ee.Filter.eq('ADM1_NAME', 'Madhya Pradesh'));
print("MP Districts", mpDistricts.size());

// Get the unified outer boundary geometry of MP to safely filter the image collection
var mpBounds = mpDistricts.geometry();

// 2. Load MODIS LST (8-day Land Surface Temperature 1km)
var modis = ee.ImageCollection("MODIS/061/MOD11A2")
              .filterDate('2017-01-01', '2026-01-01')
              .filterBounds(mpBounds); 
print("Modis",modis.size());

// Cloud/Quality masking function for both Day and Night bands
function maskMODISclouds(image) {
  var qaDay = image.select('QC_Day');
  var maskDay = qaDay.bitwiseAnd(3).lte(1);
  
  var qaNight = image.select('QC_Night');
  var maskNight = qaNight.bitwiseAnd(3).lte(1);
  
  var lstDayCelsius = image.select('LST_Day_1km')
                           .updateMask(maskDay)
                           .multiply(0.02).subtract(273.15).rename('lst');
                           
  var lstNightCelsius = image.select('LST_Night_1km')
                             .updateMask(maskNight)
                             .multiply(0.02).subtract(273.15).rename('night_lst');
  
  return image.addBands([lstDayCelsius, lstNightCelsius]).select(['lst', 'night_lst']);
}

var modisWithLST = modis.map(maskMODISclouds);
print("LST Images", modisWithLST.size());

// 3. Nested loop to map through years and months
var years = ee.List.sequence(2017, 2025);
var months = ee.List.sequence(1, 12);

var monthlyCollection = ee.ImageCollection.fromImages(
  years.map(function(y) {
    return months.map(function(m) {

      var filtered = modisWithLST
        .filter(ee.Filter.calendarRange(y, y, 'year'))
        .filter(ee.Filter.calendarRange(m, m, 'month'));

      return filtered.median()
        .set('year', y)
        .set('month', m)
        .set('system:time_start',
          ee.Date.fromYMD(y, m, 1).millis());
    });
  }).flatten()
);

// 4. Spatial aggregation over the individual districts
var districtData = monthlyCollection.map(function(image) {
  var year = image.get('year');
  var month = image.get('month');
  
  return image.reduceRegions({
    collection: mpDistricts, // Uses the separate district polygons to extract data per row
    reducer: ee.Reducer.mean(), 
    scale: 1000 
  }).map(function(feature) {
    return ee.Feature(null, {
      'district_code': feature.get('ADM2_CODE'), 
      'district': feature.get('ADM2_NAME'),
      'year': year,
      'month': month,
      'lst': feature.get('lst'), 
      'night_lst': feature.get('night_lst') 
    });
  });
}).flatten();

// 5. Clean up missing values
var outputData = districtData.filter(ee.Filter.notNull(['district', 'lst', 'night_lst']));

print("Monthly Collection Size", monthlyCollection.size());

print("District Data Size", districtData.size());

print("Output Data Size", outputData.size());

print("Sample District Data", districtData.first());

print("Sample Output Data", outputData.first());

// 6. Export to Google Drive
Export.table.toDrive({
  collection: outputData,
  description: 'MP_Districts_MODIS_Monthly_LST_Final_Fixed',
  fileFormat: 'CSV',
  selectors: ['district_code', 'district', 'year', 'month', 'lst', 'night_lst']
});