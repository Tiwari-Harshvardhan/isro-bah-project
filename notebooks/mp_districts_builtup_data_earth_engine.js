var districts =
  ee.FeatureCollection("FAO/GAUL/2015/level2");

var mpDistricts =
  districts.filter(
    ee.Filter.eq('ADM1_NAME', 'Madhya Pradesh')
  );

print("MP District Count", mpDistricts.size());

var ghsl =
  ee.ImageCollection("JRC/GHSL/P2023A/GHS_BUILT_S");

print("GHSL Image Count", ghsl.size());


// -----------------------
// Process a single year
// -----------------------

function extractYear(year) {

  var image = ee.Image(
    ghsl.filter(
      ee.Filter.eq(
        'system:index',
        String(year)
      )
    ).first()
  );

  var stats =
    image.select('built_surface')
      .reduceRegions({
        collection: mpDistricts,
        reducer: ee.Reducer.sum(),
        scale: 100,
        tileScale: 4
      });

  return stats.map(function(feature) {

    var areaSqM =
      ee.Number(feature.get('sum'));

    return ee.Feature(null, {
      district_code:
        feature.get('ADM2_CODE'),

      district:
        feature.get('ADM2_NAME'),

      year:
        year,

      built_up_area_sq_m:
        areaSqM,

      built_up_area_sq_km:
        areaSqM.divide(1000000)
    });

  });

}


// -----------------------
// Merge years
// -----------------------

var fc2015 = extractYear(2015);
var fc2020 = extractYear(2020);
var fc2025 = extractYear(2025);

var outputData =
  fc2015
    .merge(fc2020)
    .merge(fc2025)
    .filter(
      ee.Filter.notNull(
        ['built_up_area_sq_km']
      )
    );


// -----------------------
// Validation
// -----------------------

print(
  "Output Rows",
  outputData.size()
);

print(
  "Sample Row",
  outputData.first()
);

print(
  "First Five",
  outputData.limit(5)
);


// -----------------------
// Export
// -----------------------

Export.table.toDrive({
  collection: outputData,
  description:
    'MP_Districts_GHSL_BuiltUp_2015_2020_2025',
  fileFormat: 'CSV',
  selectors: [
    'district_code',
    'district',
    'year',
    'built_up_area_sq_m',
    'built_up_area_sq_km'
  ]
});