import os
from utils import load_trained_model, create_data_generators
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model_path, dataset_dir, target_size=(224, 224), batch_size=32):
    # Load model
    model = load_trained_model(model_path)

    # Create data generators for validation data
    train_path = os.path.join(dataset_dir, "train")
    val_path = os.path.join(dataset_dir, "val")

    if not os.path.exists(val_path):
        raise FileNotFoundError(f"❌ Validation directory not found: {val_path}")

    _, val_gen = create_data_generators(train_path, val_path, target_size=target_size, batch_size=batch_size)

    # Evaluate
    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f"\n✅ Model Evaluation Complete\nAccuracy: {acc:.4f}\nLoss: {loss:.4f}")

    # Predictions
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_gen.classes
    labels = list(val_gen.class_indices.keys())

    print("\nClassification Report:\n", classification_report(y_true, y_pred, target_names=labels))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Automatically detect paths based on your project structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
    MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_disease_model_best.h5")
    DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")

    print(f"Using model from: {MODEL_PATH}")
    print(f"Using dataset from: {DATASET_DIR}")

    evaluate_model(MODEL_PATH, DATASET_DIR)
