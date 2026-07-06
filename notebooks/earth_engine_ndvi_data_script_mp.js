var districts = ee.FeatureCollection("FAO/GAUL/2015/level2");
var mpDistricts = districts.filter(ee.Filter.eq('ADM1_NAME', 'Madhya Pradesh'));

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate('2018-01-01','2026-01-01')
  .filterBounds(mpDistricts);

function maskS2clouds(image) {
  var qa = image.select('QA60');
  var cloudBitMask = 1 << 10;
  var cirrusBitMask = 1 << 11;
  var mask =
    qa.bitwiseAnd(cloudBitMask).eq(0).and(qa.bitwiseAnd(cirrusBitMask).eq(0));
  return image.updateMask(mask).divide(10000);
}

var ndviCollection =s2.map(maskS2clouds).map(function(image) {
      var ndvi =image.normalizedDifference(['B8', 'B4']).rename('ndvi');
      return ndvi.copyProperties(image, ['system:time_start']
      );
    });

var years = ee.List.sequence(2018, 2025);

var months = ee.List.sequence(1, 12);

var monthlyCollection = ee.ImageCollection.fromImages(
    years.map(function(y) {return months.map(function(m) {
        var filtered = ndviCollection.filter( ee.Filter.calendarRange(y, y, 'year')).filter( ee.Filter.calendarRange(
                m, m, 'month'));
        return filtered.median().rename('ndvi').set({year: y, month: m, system_time:
              ee.Date.fromYMD(y, m, 1).millis()
          });
      });
    }).flatten()
  );

var districtData = monthlyCollection.map(function(image) {
    var year = image.get('year');
    var month = image.get('month');
    return image.reduceRegions({
      collection: mpDistricts,
      reducer: ee.Reducer.mean(),
      scale: 1000,
      tileScale: 4
    })
    .map(function(feature) {
      return ee.Feature(null, {
        district_code: feature.get('ADM2_CODE'),
        district: feature.get('ADM2_NAME'),
        year: year, month: month,
        ndvi: feature.get('mean')
      });
    });
  }).flatten();

var outputData =
  districtData.filter(ee.Filter.notNull(['district', 'ndvi']));

print("Raw S2 images", s2.size());
print("NDVI images", ndviCollection.size());
print("First NDVI image", ndviCollection.first());
print(monthlyCollection.first());

Export.table.toDrive({collection: outputData,
  description: 'MP_Districts_Sentinel2_Monthly_NDVI_2018_2025',
  fileFormat:'CSV',
  selectors: [
    'district_code',
    'district',
    'year',
    'month',
    'ndvi'

  ]

});