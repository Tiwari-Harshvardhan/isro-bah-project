import geopandas as gpd
gdf = gpd.read_file('delhi_colonies.shp')
print(gdf.head())