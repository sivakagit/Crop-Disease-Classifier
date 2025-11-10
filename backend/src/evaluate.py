import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils import load_trained_model
from utils import enable_tf_warnings
enable_tf_warnings()


def evaluate_model(model_path, dataset_dir, target_size=(224, 224), batch_size=32):
    # ---------------- LOAD TRAINED MODEL ---------------- #
    model = load_trained_model(model_path)

    # ---------------- LOAD CLASS MAPPING ---------------- #
    model_dir = os.path.dirname(model_path)
    class_indices_path = os.path.join(model_dir, "class_indices.json")

    if not os.path.exists(class_indices_path):
        raise FileNotFoundError(f"❌ Missing {class_indices_path} — cannot align labels correctly!")

    with open(class_indices_path, "r") as f:
        class_indices = json.load(f)

    index_to_class = {v: k for k, v in class_indices.items()}
    target_names = [index_to_class[i] for i in sorted(index_to_class.keys())]
    print(f"🧾 Loaded class mapping ({len(target_names)} classes) from: {class_indices_path}")

    # ---------------- CREATE VALIDATION GENERATOR ---------------- #
    val_path = os.path.join(dataset_dir, "val")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"❌ Validation directory not found: {val_path}")

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_gen = val_datagen.flow_from_directory(
        val_path,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=target_names  # 🔥 ensures same class order as training
    )

    # ---------------- EVALUATE MODEL ---------------- #
    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f"\n✅ Model Evaluation Complete\nAccuracy: {acc:.4f}\nLoss: {loss:.4f}")

    # ---------------- MAKE PREDICTIONS ---------------- #
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes

    # ---------------- CLASSIFICATION REPORT ---------------- #
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=target_names))

    # ---------------- CONFUSION MATRIX ---------------- #
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(target_names)), target_names, rotation=45, ha="right")
    plt.yticks(range(len(target_names)), target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # ---------------- AUTO-DETECT PATHS ---------------- #
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
    MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_disease_model_best.h5")
    DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")

    print(f"📁 Using model from: {MODEL_PATH}")
    print(f"📂 Using dataset from: {DATASET_DIR}\n")

    evaluate_model(MODEL_PATH, DATASET_DIR)
