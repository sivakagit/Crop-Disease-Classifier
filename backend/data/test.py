import os

root = "../data/processed"
for split in ["train", "val", "test"]:
    print(split, "->", len(os.listdir(os.path.join(root, split))), "classes")
