from __future__ import annotations

import json
from typing import Any

import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import Point
from shapely.ops import unary_union

from backend.config import DATASET_PATH, SHAPEFILE_PATH
from backend.services.csv_service import CSVService


class GeoService:
    def __init__(self, shapefile_path: str | None = None) -> None:
        self.shapefile_path = shapefile_path or str(SHAPEFILE_PATH)
        self._gdf = None
        self._zone_mapping = None

    def load_geodata(self) -> gpd.GeoDataFrame:
        if self._gdf is None:
            gdf = gpd.read_file(self.shapefile_path)
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            self._gdf = gdf
        return self._gdf

    def get_zone_names(self) -> list[str]:
        csv_service = CSVService(DATASET_PATH)
        dataset = csv_service.load_dataset()
        zone_names = [str(value).strip() for value in dataset["zone"].dropna().unique() if str(value).strip()]
        return sorted(set(zone_names))

    def _projected_point(self, longitude: float, latitude: float) -> Point:
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        x, y = transformer.transform(longitude, latitude)
        return Point(x, y)

    def locate_zone_from_coordinates(self, longitude: float, latitude: float) -> str | None:
        gdf = self.load_geodata()
        projected = gdf.to_crs(epsg=3857)
        point = self._projected_point(longitude, latitude)
        candidates = projected[projected.geometry.contains(point)]
        if not candidates.empty:
            zone_id = int(candidates.iloc[0]["zone"])
            zone_names = self.get_zone_names()
            return zone_names[zone_id - 1] if 0 < zone_id <= len(zone_names) else str(zone_id)
        return None

    def select_geometry(self, longitude: float, latitude: float) -> dict[str, Any]:
        gdf = self.load_geodata()
        projected = gdf.to_crs(epsg=3857)
        point_projected = self._projected_point(longitude, latitude)
        candidates = projected[projected.geometry.contains(point_projected)]
        if candidates.empty:
            candidates = projected[projected.geometry.buffer(50).contains(point_projected)]
        if candidates.empty:
            centroid_candidates = projected[projected.geometry.centroid.buffer(50).contains(point_projected)]
            if not centroid_candidates.empty:
                candidates = centroid_candidates
        if candidates.empty:
            return {"zone": None, "ward": None, "geometry": None, "zone_name": None}
        selected = candidates.iloc[0]
        geometry_data = selected.geometry.__geo_interface__
        if isinstance(geometry_data, dict):
            geometry_payload = geometry_data
        else:
            geometry_payload = json.loads(geometry_data)
        return {
            "zone": str(selected.get("zone", "")),
            "ward": str(selected.get("ward", "")),
            "zone_name": str(selected.get("zone", "")),
            "geometry": geometry_payload,
        }

    def get_map_geojson(self) -> dict[str, Any]:
        gdf = self.load_geodata()
        if self._zone_mapping is None:
            self.get_zone_names()
        gdf = gdf.copy()
        gdf["zone_name"] = gdf["zone"].astype(int).map(self._zone_mapping)
        return json.loads(gdf.to_json())
