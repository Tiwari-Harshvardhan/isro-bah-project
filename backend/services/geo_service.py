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
            if gdf.crs is None:
                gdf = gdf.set_crs(epsg=4326, allow_override=True)
            self._gdf = gdf
        return self._gdf

    def _build_zone_name_mapping(self) -> dict[int, str]:
        if self._zone_mapping is not None:
            return self._zone_mapping

        dataset = self.csv_service.load_dataset()
        dataset = dataset.dropna(subset=["zone", "ward"]).copy()
        zone_names = sorted(dataset["zone"].astype(str).unique())
        ward_sets = {
            zone_name: set(dataset[dataset["zone"] == zone_name]["ward"].astype(str).str.strip().str.lower().unique())
            for zone_name in zone_names
        }

        gdf = self.load_geodata().copy()
        gdf["zone_id"] = gdf["zone"].astype(int)
        gdf["area_norm"] = gdf["area"].astype(str).str.strip().str.lower()

        mapping: dict[int, str] = {}
        for zone_id in sorted(gdf["zone_id"].unique()):
            zone_area_names = set(gdf[gdf["zone_id"] == zone_id]["area_norm"].dropna().unique())
            best_match = None
            best_score = 0
            for zone_name, ward_set in ward_sets.items():
                overlap = len(zone_area_names & ward_set)
                if overlap > best_score:
                    best_score = overlap
                    best_match = zone_name
            if best_match is not None and best_score > 0:
                mapping[zone_id] = best_match
            else:
                mapping[zone_id] = f"Zone-{zone_id}"

        self._zone_mapping = mapping
        return self._zone_mapping

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
