import os
import numpy as np

INPUT_DIR = r"C:\Users\omsai\6th-sem Projects\Remote Sensing\features\final"

X = []
y = []

files = os.listdir(INPUT_DIR)

for f in files:
    if not f.endswith(".npy"):
        continue

    data = np.load(os.path.join(INPUT_DIR, f))  # (H, W, 5)

    pixels = data.reshape(-1, 5)

    # -------------------------------
    # PSEUDO LABELS (INITIAL)
    # -------------------------------
    slope = pixels[:, 3]
    amplitude = pixels[:, 2]

    labels = ((slope < -0.0005) & (amplitude < 0.1)).astype(int)

    X.append(pixels)
    y.append(labels)

X = np.vstack(X)
y = np.hstack(y)

print("Dataset shape:", X.shape, y.shape)
print("Class distribution:")
print("0:", np.sum(y == 0))
print("1:", np.sum(y == 1))