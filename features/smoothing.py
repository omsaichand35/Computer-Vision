import numpy as np
import os
from scipy.signal import savgol_filter

INPUT_DIR = r"C:\Users\omsai\6th-sem Projects\Remote Sensing\time_series"
OUTPUT_DIR = r"C:\Users\omsai\6th-sem Projects\Remote Sensing\features\smoothed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = os.listdir(INPUT_DIR)

for f in files:
    if not f.endswith(".npy"):
        continue

    data = np.load(os.path.join(INPUT_DIR, f))  # (T, H, W)

    T, H, W = data.shape

    smoothed = np.zeros_like(data)

    # Apply smoothing pixel-wise
    for i in range(H):
        for j in range(W):
            try:
                smoothed[:, i, j] = savgol_filter(
                    data[:, i, j],
                    window_length=11,   # must be odd
                    polyorder=2
                )
            except:
                smoothed[:, i, j] = data[:, i, j]

    np.save(os.path.join(OUTPUT_DIR, f), smoothed)

    print(f"✅ Smoothed {f}")

print("🔥 All time-series smoothed!")