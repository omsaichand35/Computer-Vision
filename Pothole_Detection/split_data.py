import os
import shutil
import random

source_path = "../pothole"
output_path = "../pothole_dataset"

classes = ["normal", "potholes"]
split_ratio = 0.8

for cls in classes:
    src = os.path.join(source_path, cls)
    images = os.listdir(src)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train = images[:split_index]
    test = images[split_index:]

    for img in train:
        dst_dir = os.path.join(output_path, "train", cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(os.path.join(src, img), os.path.join(dst_dir, img))

    for img in test:
        dst_dir = os.path.join(output_path, "val", cls)
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(os.path.join(src, img), os.path.join(dst_dir, img))

print("Dataset split complete.")