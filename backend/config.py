import logging
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODEL_DIR = BACKEND_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
DATASET_PATH = DATA_DIR / "processed_dataset.csv"
SHAPEFILE_PATH = BASE_DIR / "data" / "raw" / "delhicolonies" / "delhi_colonies.shp"

LOG_LEVEL = os.getenv("URBANCOOL_LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("urbancool")
