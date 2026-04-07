import os
import shutil

# 🔧 CHANGE THIS PATH (your current data root)
OLD_ROOT = r"D:/processed_data"

# 🔧 NEW PROJECT ROOT
NEW_ROOT = r"C:\Users\omsai\6th-sem Projects\Remote Sensing"

# -------------------------------
# STEP 1: CREATE NEW STRUCTURE
# -------------------------------

folders = [
    "raw",
    "interim/ndvi",
    "interim/qa",
    "interim/evi",
    "processed/clean_ndvi",
    "processed/scaled_ndvi",
    "tiles",
    "time_series",
    "features",
    "labels"
]

for f in folders:
    os.makedirs(os.path.join(NEW_ROOT, f), exist_ok=True)

print("✅ Folder structure created")


# -------------------------------
# STEP 2: MOVE NDVI + QA
# -------------------------------

def move_files(src_folder, dest_folder):
    if not os.path.exists(src_folder):
        return

    for file in os.listdir(src_folder):
        src = os.path.join(src_folder, file)
        dst = os.path.join(dest_folder, file)

        if os.path.isfile(src):
            shutil.move(src, dst)


# Move NDVI
move_files(os.path.join(OLD_ROOT, "ndvi"),
           os.path.join(NEW_ROOT, "interim/ndvi"))

# Move QA
move_files(os.path.join(OLD_ROOT, "qa"),
           os.path.join(NEW_ROOT, "interim/qa"))

# Move cleaned + scaled
move_files(os.path.join(OLD_ROOT, "clean_ndvi"),
           os.path.join(NEW_ROOT, "processed/clean_ndvi"))

move_files(os.path.join(OLD_ROOT, "scaled_ndvi"),
           os.path.join(NEW_ROOT, "processed/scaled_ndvi"))

print("✅ Files moved")

# -------------------------------
# STEP 3: RENAME TILE FOLDERS (CLEAN TIMESTAMP)
# -------------------------------

old_tiles = os.path.join(OLD_ROOT, "tiles")
new_tiles = os.path.join(NEW_ROOT, "tiles")

os.makedirs(new_tiles, exist_ok=True)

for folder in os.listdir(old_tiles):
    old_path = os.path.join(old_tiles, folder)

    if os.path.isdir(old_path):
        # Extract date (A2016065 → 2016065)
        parts = folder.split(".")

        date_part = None
        for p in parts:
            if p.startswith("A") and len(p) == 8:
                date_part = p[1:]
                break

        if date_part is None:
            print(f"⚠️ Skipping {folder}")
            continue

        new_path = os.path.join(new_tiles, date_part)

        shutil.copytree(old_path, new_path)

        print(f"✔ {folder} → {date_part}")

print("✅ Tiles renamed and moved")

# -------------------------------
# DONE
# -------------------------------

print("\n🔥 STRUCTURE READY FOR TIME-SERIES!")