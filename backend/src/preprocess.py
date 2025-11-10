import os
import random
from pathlib import Path
from PIL import Image
import shutil

# ---------------- CONFIG ---------------- #
BASE_DIR = Path(__file__).resolve().parent.parent  # E:/Crop-disease-classifier
RAW_DIR = BASE_DIR / "data" / "PlantVillage"       # Source images
OUT_DIR = BASE_DIR / "data" / "dataset"                     # Output folder (completely separate!)

CLASSES = [
    'Tomato___Late_blight',
    'Tomato___Early_blight',
    'Tomato___Leaf_Mold',
    'Tomato___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy'
]

IMG_SIZE = (224, 224)
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
random.seed(42)

# ---------------- CREATE FOLDERS ---------------- #
def create_dirs():
    for split in ['train', 'val', 'test']:
        for cls in CLASSES:
            path = OUT_DIR / split / cls
            path.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output folders ready at: {OUT_DIR}")

# ---------------- IMAGE PROCESSING ---------------- #
def process_images():
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
            # Save resized copy (not overwrite!)
            img.save(out_path)

        # ✅ Optional: verify original folder remains untouched
        print(f"✅ Processed class: {cls} | Original images preserved.")

    print("\n✅ Dataset processed and split safely into train/val/test!")
    print(f"📂 Final output at: {OUT_DIR.resolve()}")

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    process_images()
