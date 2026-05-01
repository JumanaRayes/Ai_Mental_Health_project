from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = BASE_DIR / "AIModels" / "saved_models" / "risk"
TOKENIZER_PATH = BASE_DIR / "AIModels" / "saved_models" / "risk" / "tokenizer.pkl"


print("FILE LOCATION:", __file__)
print("DIRNAME:", os.path.dirname(__file__))
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
