import os
import shutil
import random

source_dir = 'dataset/cyaug dataset'
target_dir = 'dataset'
train_ratio = 0.8  # 80% training, 20% validation

soil_classes = os.listdir(source_dir)

for cls in soil_classes:
    class_path = os.path.join(source_dir, cls)
    images = os.listdir(class_path)
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:]

    os.makedirs(os.path.join(target_dir, 'train', cls), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'val', cls), exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(target_dir, 'train', cls, img))
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(target_dir, 'val', cls, img))

print("✅ Dataset split into train/ and val/")
