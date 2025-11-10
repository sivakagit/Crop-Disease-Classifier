import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf

# ==========================================================
# ⚙️ CONFIG
# ==========================================================
MODEL_PATH = os.path.join("backend", "models", "crop_disease_model_best.tflite")
IMG_SIZE = (224, 224)  # must match your training input size

# ==========================================================
# 🚀 FLASK APP INIT
# ==========================================================
app = Flask(__name__)

# ==========================================================
# 🧠 LOAD TFLITE MODEL (lightweight)
# ==========================================================
print("🚀 Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite model loaded successfully!")

# ==========================================================
# 🧮 PREDICTION ENDPOINT
# ==========================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        img = Image.open(file.stream).convert("RGB")
        img = img.resize(IMG_SIZE)
        img = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])[0]

        # Convert predictions to readable output
        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return jsonify({
            "predicted_class": predicted_index,
            "confidence": round(confidence * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================================================
# 🏠 HOME ENDPOINT
# ==========================================================
@app.route("/", methods=["GET"])
def home():
    return (
        "<h2>🌿 Crop Disease Classifier API (TFLite)</h2>"
        "<p>Use POST /predict with an image file.</p>"
    )


# ==========================================================
# 🧩 ENTRY POINT (Render/Gunicorn)
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
