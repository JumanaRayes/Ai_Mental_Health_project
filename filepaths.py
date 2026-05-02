from pathlib import Path
import os

# 🔹 Get project root safely (no hardcoded depth)
BASE_DIR = Path(__file__).resolve().parent

# 🔹 Paths
MODEL_PATH = BASE_DIR / "AIModels" / "saved_models" / "risk"
TOKENIZER_PATH = MODEL_PATH / "tokenizer.pkl"

# 🔹 Debug (optional - remove later)
print("FILE LOCATION:", __file__)
print("BASE_DIR:", BASE_DIR)
print("MODEL_PATH:", MODEL_PATH)
print("TOKENIZER_PATH:", TOKENIZER_PATH)

# 🔹 Validation (very useful)
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model folder not found: {MODEL_PATH}")

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_PATH}")

