import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parent
BASE_DIR = BACKEND_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
DATASET_PATH = DATA_DIR / "processed_dataset.csv"
SHAPEFILE_PATH = BASE_DIR / "data" / "raw" / "delhicolonies" / "delhi_colonies.shp"

load_dotenv(BASE_DIR / ".env")

LOG_LEVEL = os.getenv("URBANCOOL_LOG_LEVEL", "INFO").upper()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("urbancool")
