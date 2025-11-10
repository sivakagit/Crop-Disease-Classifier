import os
import json
import random
from pathlib import Path
from PIL import Image

# ---------------- CONFIG ---------------- #
BASE_DIR = Path(__file__).resolve().parent.parent  # e.g., E:/Crop-disease-classifier/backend
RAW_DIR = BASE_DIR / "data" / "PlantVillage"       # Source dataset
OUT_DIR = BASE_DIR / "data" / "dataset"            # Output processed dataset

IMG_SIZE = (224, 224)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
random.seed(42)

# ---------------- AUTO-DETECT CLASSES ---------------- #
CLASSES = [d.name for d in RAW_DIR.iterdir() if d.is_dir()]
print(f"✅ Detected {len(CLASSES)} classes: {CLASSES}")

# ---------------- CREATE FOLDERS ---------------- #
def create_dirs():
    """Creates train/val/test directories for all classes."""
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            path = OUT_DIR / split / cls
            path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output folders ready at: {OUT_DIR}")

# ---------------- IMAGE PROCESSING ---------------- #
def process_images():
    """Resizes and splits PlantVillage dataset into train/val/test."""
    create_dirs()
    for cls in CLASSES:
        src_folder = RAW_DIR / cls
        if not src_folder.exists():
            print(f"⚠️ Missing folder {cls}, skipping.")
            continue

        imgs = list(src_folder.glob('*'))
        if len(imgs) == 0:
            print(f"⚠️ No images found in {cls}")
            continue

        random.shuffle(imgs)
        n = len(imgs)
        n_train = int(n * TRAIN_SPLIT)
        n_val = int(n * VAL_SPLIT)

        for i, img_path in enumerate(imgs):
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize(IMG_SIZE)
            except Exception as e:
                print(f"❌ Skipped unreadable image {img_path}: {e}")
                continue

            if i < n_train:
                split = 'train'
            elif i < n_train + n_val:
                split = 'val'
            else:
                split = 'test'

            out_path = OUT_DIR / split / cls / img_path.name
            img.save(out_path)

        print(f"✅ Processed class: {cls} | {n} images resized & split.")

    print("\n🎉 Dataset processed and split safely into train/val/test!")
    print(f"📂 Final output at: {OUT_DIR.resolve()}")

    # Save the class names for reference
    labels_path = OUT_DIR / "class_names.json"
    with open(labels_path, "w") as f:
        json.dump(CLASSES, f, indent=4)
    print(f"🧾 Saved class names to: {labels_path}")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    process_images()
