import os
import numpy as np
from scipy.stats import linregress

INPUT_DIR = r"C:\Users\omsai\6th-sem Projects\Remote Sensing\features\smoothed"
OUTPUT_DIR = r"C:\Users\omsai\6th-sem Projects\Remote Sensing\features\final"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = os.listdir(INPUT_DIR)

for f in files:
    if not f.endswith(".npy"):
        continue

    data = np.load(os.path.join(INPUT_DIR, f))  # (T, H, W)
    T, H, W = data.shape

    # Flatten spatial
    pixels = data.reshape(T, -1)  # (T, N_pixels)

    # -------------------------------
    # FEATURE 1: MEAN NDVI
    # -------------------------------
    mean_ndvi = np.mean(pixels, axis=0)

    # -------------------------------
    # FEATURE 2: VARIANCE
    # -------------------------------
    variance = np.var(pixels, axis=0)

    # -------------------------------
    # FEATURE 3: AMPLITUDE
    # -------------------------------
    amplitude = np.max(pixels, axis=0) - np.min(pixels, axis=0)

    # -------------------------------
    # FEATURE 4: TREND (SLOPE)
    # -------------------------------
    slopes = []

    time = np.arange(T)

    for i in range(pixels.shape[1]):
        y = pixels[:, i]

        if np.all(np.isnan(y)):
            slopes.append(0)
            continue

        slope, _, _, _, _ = linregress(time, y)
        slopes.append(slope)

    slopes = np.array(slopes)

    # -------------------------------
    # FEATURE 5: DROP DETECTION
    # -------------------------------
    diff = np.diff(pixels, axis=0)
    sudden_drop = np.min(diff, axis=0)

    # -------------------------------
    # STACK FEATURES
    # -------------------------------
    features = np.stack([
        mean_ndvi,
        variance,
        amplitude,
        slopes,
        sudden_drop
    ], axis=1)  # (N_pixels, 5)

    # reshape back to image grid
    features = features.reshape(H, W, 5)

    np.save(os.path.join(OUTPUT_DIR, f), features)

    print(f"✅ Features extracted for {f}")

print("\n🔥 ALL FEATURES EXTRACTED!")