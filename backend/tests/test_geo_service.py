from __future__ import annotations

import geopandas as gpd

from backend.services.geo_service import GeoService


def test_select_geometry_matches_known_shapefile_coordinate() -> None:
    shapefile_path = r"C:/Users/harsh/Desktop/isro hacakthon/data/raw/delhicolonies/delhi_colonies.shp"
    gdf = gpd.read_file(shapefile_path)
    row = gdf.iloc[0]

    service = GeoService(shapefile_path=shapefile_path)
    result = service.select_geometry(float(row["longitude"]), float(row["latitude"]))

    assert result["zone"] is not None
    assert result["ward"] is not None
