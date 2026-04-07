import os
import numpy as np
import joblib

MODEL_PATH = "model.pkl"
INPUT_DIR = r"C:\Users\omsai\6th-sem Projects\Remote Sensing\features\final"
OUTPUT_DIR = r"C:\Users\omsai\6th-sem Projects\Remote Sensing\outputs\ml_maps"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = joblib.load(MODEL_PATH)

files = os.listdir(INPUT_DIR)

for f in files:
    if not f.endswith(".npy"):
        continue

    data = np.load(os.path.join(INPUT_DIR, f))  # (H, W, 5)

    H, W, _ = data.shape

    pixels = data.reshape(-1, 5)

    preds = model.predict(pixels)

    preds = preds.reshape(H, W)

    np.save(os.path.join(OUTPUT_DIR, f), preds)

    print(f"✅ ML map: {f}")