from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "year",
    "year_index",
    "month",
    "population",
    "population_density",
    "area_sq_km",
    "built_up_percent",
    "mean_ndvi",
    "ndvi_lag1",
    "ndvi_lag2",
    "ndvi_3month_avg",
    "ndvi_change",
    "lst_3month_avg",
    "lst_12month_avg",
    "builtup_ndvi",
    "urban_intensity",
    "heat_exposure_index",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for column in FEATURE_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    if "year_index" not in data.columns:
        data["year_index"] = data["year"] - data["year"].min()

    for column in ["ndvi_lag1", "ndvi_lag2", "ndvi_change"]:
        if column in data.columns:
            data[column] = data.groupby("zone")[column].ffill()
            data[column] = data[column].fillna(0)

    for column in ["ndvi_3month_avg", "lst_3month_avg", "lst_12month_avg"]:
        if column in data.columns:
            data[column] = data.groupby("zone")[column].ffill()
            data[column] = data[column].fillna(data[column].mean())

    if "builtup_ndvi" in data.columns:
        data["builtup_ndvi"] = data["builtup_ndvi"].fillna(data["built_up_percent"] * data["mean_ndvi"])

    for column in ["urban_intensity", "heat_exposure_index"]:
        if column in data.columns:
            data[column] = data.groupby("zone")[column].ffill()
            data[column] = data[column].fillna(data[column].mean())

    return data[FEATURE_COLUMNS]
