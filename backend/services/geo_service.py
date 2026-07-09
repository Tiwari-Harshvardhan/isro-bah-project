from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point

from backend.config import DATASET_PATH, SHAPEFILE_PATH
from backend.services.csv_service import CSVService


class GeoService:
    def __init__(self, shapefile_path: str | None = None) -> None:
        self.shapefile_path = shapefile_path or str(SHAPEFILE_PATH)
        self._gdf = None
        self._zone_mapping: dict[int, str] | None = None
        self.csv_service = CSVService()

    def load_geodata(self) -> gpd.GeoDataFrame:
        if self._gdf is None:
            shapefile_path = Path(self.shapefile_path)
            if not shapefile_path.exists():
                alt_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "delhicolonies" / "delhi_colonies.shp"
                if alt_path.exists():
                    shapefile_path = alt_path
                else:
                    raise FileNotFoundError(
                        f"Shapefile not found at {self.shapefile_path} or alternate path {alt_path}"
                    )
            gdf = gpd.read_file(shapefile_path)
            gdf["geometry"] = gdf.geometry.make_valid()
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            self._gdf = gdf
        return self._gdf

    def _build_zone_name_mapping(self) -> dict[int, str]:
        return {
            1: "Keshav Puram Zone",
            2: "Shahdara South Zone",
            3: "South Zone",
            4: "City-Sadar Paharganj(SP) Zone",
            5: "City-Sadar Paharganj(SP) Zone",
            6: "Civil Lines Zone",
            7: "Narela Zone",
            8: "Rohini Zone",
            9: "Central-Zone",
            10: "West Zone",
            11: "Karol Bagh Zone",
            12: "Karol Bagh Zone",
        }

    def get_zone_names(self) -> list[str]:
        dataset = self.csv_service.load_dataset()
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
            return self._build_zone_name_mapping().get(zone_id)
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
        zone_id = int(selected.get("zone", 0))
        zone_name = self._build_zone_name_mapping().get(zone_id, f"Zone-{zone_id}")
        geometry_payload = selected.geometry.__geo_interface__ if isinstance(selected.geometry.__geo_interface__, dict) else json.loads(selected.geometry.__geo_interface__)

        return {
            "zone": str(zone_id),
            "ward": str(selected.get("ward", "")),
            "zone_name": zone_name,
            "geometry": geometry_payload,
        }

    def get_map_geojson(self) -> dict[str, Any]:
        gdf = self.load_geodata().copy()
        mapping = self._build_zone_name_mapping()
        gdf["zone_name"] = gdf["zone"].astype(int).map(mapping).fillna("Unknown Zone")
        gdf = gdf.to_crs(epsg=4326)
        dissolved = gdf.dissolve(by="zone_name", as_index=False)
        return json.loads(dissolved.to_json())
