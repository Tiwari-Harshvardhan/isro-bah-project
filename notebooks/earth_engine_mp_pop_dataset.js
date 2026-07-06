var districts =
  ee.FeatureCollection("FAO/GAUL/2015/level2");

var mpDistricts =
  districts.filter(
    ee.Filter.eq('ADM1_NAME', 'Madhya Pradesh')
  );

print("MP District Count", mpDistricts.size());

// Use the GHSL Population dataset
var ghslPop =
  ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP");

print("GHSL Population Image Count", ghslPop.size());

// ========================================
// VALIDATION CHECK 1: Verify band names
// ========================================
var sampleImage = ee.Image(ghslPop.first());

print("Population Band Names", sampleImage.bandNames());
print("Population First Image", sampleImage);

// ========================================
// VALIDATION CHECK 2: Verify available years
// ========================================
print(
  "Available Population Years",
  ghslPop.aggregate_array('system:index')
);

// ========================================
// VALIDATION CHECK 3: Check projection
// ========================================
print(
  "Population Projection",
  sampleImage.projection()
);

// -----------------------
// Process a single year
// -----------------------

function extractYear(year) {

  var image = ee.Image(
    ghslPop.filter(
      ee.Filter.eq(
        'system:index',
        String(year)
      )
    ).first()
  );

  var stats =
    image.select('population_count')
      .reduceRegions({
        collection: mpDistricts,
        reducer: ee.Reducer.sum(),
        scale: 100,  // Native resolution of GHSL-POP
        tileScale: 4
      });

  return stats.map(function(feature) {

    var population =
      ee.Number(feature.get('sum'));

    return ee.Feature(null, {
      district_code:
        feature.get('ADM2_CODE'),

      district:
        feature.get('ADM2_NAME'),

      year:
        year,

      total_population:
        population
    });

  });

}

// ========================================
// VALIDATION CHECK 4: Test one district manually
// ========================================
var bhopal = mpDistricts.filter(
  ee.Filter.eq('ADM2_NAME', 'Bhopal')
);

var image2025 = ee.Image(
  ghslPop.filter(
    ee.Filter.eq('system:index', '2025')
  ).first()
);

var testPopulation = image2025
  .select('population_count')
  .reduceRegion({
    reducer: ee.Reducer.sum(),
    geometry: bhopal.geometry(),
    scale: 100,
    maxPixels: 1e13
  });

print("Bhopal Population 2025 (Manual Test)", testPopulation);

// ========================================
// VALIDATION CHECK 5: Test single year
// ========================================
var test2025 = extractYear(2025);

print("2025 Rows", test2025.size());
print("2025 Sample", test2025.first());

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
        ['total_population']
      )
    );

// ========================================
// VALIDATION CHECK 6: Final sanity checks
// ========================================
print("================================");
print("SANITY CHECKS");
print("================================");

print("MP District Count", mpDistricts.size());

print(
  "Available Population Years",
  ghslPop.aggregate_array('system:index')
);

print(
  "Population Bands",
  ee.Image(ghslPop.first()).bandNames()
);

var test2025Final = extractYear(2025);

print("2025 Rows", test2025Final.size());

print(
  "2025 Sample",
  test2025Final.first()
);

print(
  "Final Output Rows",
  outputData.size()
);

print(
  "Final Sample",
  outputData.first()
);

// -----------------------
// Validation Output Summary
// -----------------------
print("================================");
print("EXPECTED RESULTS BEFORE EXPORT:");
print("================================");
print("MP District Count = 48");
print("Population Bands = ['population_count']");
print("2025 Rows = 48");
print("Final Output Rows = 144 (48 districts × 3 years)");
print("Final Sample = valid row with district, year, total_population");

// -----------------------
// Export
// -----------------------

Export.table.toDrive({
  collection: outputData,
  description:
    'MP_Districts_GHSL_Population_2015_2020_2025',
  fileFormat: 'CSV',
  selectors: [
    'district_code',
    'district',
    'year',
    'total_population'
  ]
});