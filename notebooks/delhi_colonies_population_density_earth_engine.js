// ========================================
// 1. LOAD DELHI COLONIES AND DISSOLVE INTO WARDS
// ========================================
var colonies = table;

print("Delhi Colonies Count:", colonies.size());
print("Shapefile columns:", colonies.first().propertyNames());

Map.centerObject(colonies, 11);
Map.addLayer(colonies, {color: 'red'}, 'Delhi Colonies');

// Get the distinct list of ward names to dissolve by.
var wardNames = colonies.aggregate_array('ward').distinct();
print("Distinct wards:", wardNames.size());

// Build one dissolved (unioned) polygon per ward, carrying its zone along.
// maxError=1 (metre) makes the union tolerant of small topology issues,
// similar in spirit to make_valid() on the Python side.
var wards = ee.FeatureCollection(wardNames.map(function (wardName) {
  var wardFeatures = colonies.filter(ee.Filter.eq('ward', wardName));
  var unionGeom = wardFeatures.geometry().dissolve(1);
  var zone = wardFeatures.first().get('zone');
  var areaSqKm = ee.Number(unionGeom.area(1)).divide(1e6);

  return ee.Feature(unionGeom, {
    zone: zone,
    ward: wardName,
    area_sq_km: areaSqKm
  });
}));

print("Dissolved ward count:", wards.size());
Map.addLayer(wards, {color: 'blue'}, 'Delhi Wards (dissolved)');

// ========================================
// 2. LOAD GHSL POPULATION DATASET
// ========================================
var ghslPop = ee.ImageCollection("JRC/GHSL/P2023A/GHS_POP");
print("GHSL Population Image Count:", ghslPop.size());
print("Available Population Years:", ghslPop.aggregate_array('system:index'));

// ========================================
// 3. EXTRACT REAL POPULATION FOR THE THREE ANCHOR YEARS
// ========================================
print("========================================");
print("EXTRACTING REAL ANCHOR-YEAR POPULATION (2015, 2020, 2025)");
print("========================================");

function extractAnchorYear(year) {
  var image = ghslPop.filter(ee.Filter.eq('system:index', String(year))).first();

  var summed = image.select('population_count').reduceRegions({
    collection: wards,
    reducer: ee.Reducer.sum(),
    scale: 100,
    tileScale: 8
  });

  // reduceRegions names the result 'sum'; rename to something year-specific
  // so we can carry all three anchor years as separate properties per ward.
  return summed.map(function (f) {
    var pop = ee.Number(ee.Algorithms.If(f.get('sum'), f.get('sum'), 0));
    return f.set('pop_' + year, pop);
  });
}

var anchor2015 = extractAnchorYear(2015);
var anchor2020 = extractAnchorYear(2020);
var anchor2025 = extractAnchorYear(2025);

print("2015 rows:", anchor2015.size());
print("2020 rows:", anchor2020.size());
print("2025 rows:", anchor2025.size());

// ========================================
// 4. JOIN THE THREE ANCHOR YEARS BACK ONTO A SINGLE PER-WARD FEATURE
// ========================================
var joinFilter = ee.Filter.equals({leftField: 'ward', rightField: 'ward'});
var innerJoin = ee.Join.inner();

var joined1 = innerJoin.apply(anchor2015, anchor2020, joinFilter);
var joined1Flat = joined1.map(function (pair) {
  var f1 = ee.Feature(pair.get('primary'));
  var f2 = ee.Feature(pair.get('secondary'));
  return f1.set('pop_2020', f2.get('pop_2020'));
});

var joined2 = innerJoin.apply(joined1Flat, anchor2025, joinFilter);
var wardsWithAnchors = joined2.map(function (pair) {
  var f1 = ee.Feature(pair.get('primary'));
  var f2 = ee.Feature(pair.get('secondary'));
  return f1.set('pop_2025', f2.get('pop_2025'));
});

print("Wards with all 3 anchor years joined:", wardsWithAnchors.size());
print("Sample joined ward:", wardsWithAnchors.first());

// ========================================
// 5. EXPAND TO MONTHLY ROWS, 2018-01 THROUGH 2022-12, IN CHRONOLOGICAL ORDER
// ========================================
print("========================================");
print("INTERPOLATING MONTHLY POPULATION (2018-01 .. 2022-12)");
print("========================================");

var years = ee.List.sequence(2018, 2022);
var months = ee.List.sequence(1, 12);

// value(year) interpolates linearly between the two real anchors that
// bracket it. Every year in 2018-2022 is fully bracketed by 2015/2020/2025,
// so this is true interpolation end-to-end -- no extrapolation needed here.
function valueForYear(pop2015, pop2020, pop2025, year) {
  var y = ee.Number(year);
  return ee.Number(ee.Algorithms.If(
    y.lte(2020),
    pop2015.add(pop2020.subtract(pop2015).multiply(y.subtract(2015).divide(5))),
    pop2020.add(pop2025.subtract(pop2020).multiply(y.subtract(2020).divide(5)))
  ));
}

// Nested map: for each year -> for each month -> for each ward -> one row.
// A single ee.List.flatten() call recursively collapses all nesting levels,
// and because we build it in (year -> month -> ward) order, the resulting
// list is already in strict chronological order: 2018-01, 2018-02, ... 2022-12.
var monthlyFeaturesNested = years.map(function (year) {
  var y = ee.Number(year);

  return months.map(function (month) {
    var m = ee.Number(month);
    var method = ee.String(ee.Algorithms.If(y.eq(2020), 'computed', 'interpolated'));

    var monthFC = wardsWithAnchors.map(function (ward) {
      var pop2015 = ee.Number(ward.get('pop_2015'));
      var pop2020 = ee.Number(ward.get('pop_2020'));
      var pop2025 = ee.Number(ward.get('pop_2025'));
      var areaSqKm = ee.Number(ward.get('area_sq_km'));

      var population = valueForYear(pop2015, pop2020, pop2025, y);
      var density = ee.Number(ee.Algorithms.If(
        areaSqKm.gt(0), population.divide(areaSqKm), 0
      ));

      return ee.Feature(null, {
        zone: ward.get('zone'),
        ward: ward.get('ward'),
        year: y,
        month: m,
        population: population,
        population_density: density,
        area_sq_km: areaSqKm,
        method: method
      });
    });

    // KEY FIX (same bug as before): convert to a real List<Feature>
    // BEFORE the outer .flatten() call needs to unwrap it.
    return monthFC.toList(monthFC.size());
  });
});

var monthlyFeaturesFlat = ee.FeatureCollection(monthlyFeaturesNested.flatten());
print("Total monthly rows:", monthlyFeaturesFlat.size());
print("Expected rows (60 months x wards):", wards.size().multiply(60));

// ========================================
// 6. VALIDATION - SAMPLE A FEW MONTHS
// ========================================
print("========================================");
print("VALIDATION - SAMPLE DATA");
print("========================================");

var sampleYears = [2018, 2020, 2022];
sampleYears.forEach(function (year) {
  var yearData = monthlyFeaturesFlat.filter(ee.Filter.and(
    ee.Filter.eq('year', year),
    ee.Filter.eq('month', 1)
  ));
  var totalPop = yearData.aggregate_sum('population');
  var count = yearData.size();

  totalPop.evaluate(function (totalVal) {
    count.evaluate(function (countVal) {
      print("Jan " + year + ": Total Population = " + totalVal + ", Wards = " + countVal);
    });
  });
});

// ========================================
// 7. EXPORT TO GOOGLE DRIVE
// ========================================
print("========================================");
print("EXPORTING DATA");
print("========================================");

Export.table.toDrive({
  collection: monthlyFeaturesFlat,
  description: 'Delhi_Ward_Monthly_Population_Density_2018_2022',
  fileFormat: 'CSV',
  selectors: ['zone', 'ward', 'year', 'month', 'population', 'population_density', 'area_sq_km', 'method']
});

// ========================================
// 8. SUMMARY
// ========================================
print("========================================");
print("SUMMARY");
print("========================================");
print("✓ Total wards:", wards.size());
print("✓ Monthly range: 2018-01 through 2022-12 (60 months, chronological order)");
print("✓ Real anchor years: 2015, 2020, 2025 (GHSL Population)");
print("✓ 2020 = computed (real). 2018, 2019, 2021, 2022 = interpolated (bracketed by real anchors)");
print("");
print("Export: Delhi_Ward_Monthly_Population_Density_2018_2022.csv");
print("  Columns: zone, ward, year, month, population, population_density, area_sq_km, method");
print("========================================");