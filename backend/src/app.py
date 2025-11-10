from flask import Flask, request, render_template_string, session
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import os
from werkzeug.utils import secure_filename
from utils import enable_tf_warnings

# ---------------- SETUP ---------------- #
enable_tf_warnings()
app = Flask(__name__)
app.secret_key = "crop-disease-secret"  # required for session history

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/crop_disease_model_best.h5")
LABELS_PATH = os.path.join(BASE_DIR, "../models/class_indices.json")
INFO_PATH = os.path.join(BASE_DIR, "../data/disease_info.json")

# ---------------- LOAD MODEL ---------------- #
print("🚀 Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ---------------- LOAD LABELS ---------------- #
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
index_to_label = {v: k for k, v in class_indices.items()}

# ---------------- LOAD DISEASE INFO ---------------- #
if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r") as f:
        disease_info = json.load(f)
else:
    disease_info = {}
    print("⚠️ disease_info.json not found — remedies will be empty.")

# ---------------- IMAGE PREPROCESSING ---------------- #
def prepare_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# ---------------- HTML TEMPLATE ---------------- #
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>🌿 Crop Disease Classifier</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
body {
    font-family: 'Segoe UI', sans-serif;
    background: #f3f9f5;
    text-align: center;
    padding: 50px;
}
.container {
    background: white;
    display: inline-block;
    padding: 40px;
    border-radius: 16px;
    box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    max-width: 520px;
}
h1 { color: #2e7d32; margin-bottom: 10px; }
input[type=file] { margin-top: 15px; }
img { max-width: 250px; margin-top: 15px; border-radius: 12px; display: none; }
button {
    margin-top: 20px;
    padding: 12px 24px;
    font-size: 16px;
    color: white;
    background: #2e7d32;
    border: none;
    border-radius: 8px;
    cursor: pointer;
}
button:hover { background: #256228; }

.result-card {
    margin-top: 25px;
    text-align: left;
    border-radius: 10px;
    padding: 20px;
    color: white;
}
.result-card.healthy { background: #2e7d32; }
.result-card.mild { background: #ffc107; color: #333; }
.result-card.disease { background: #d32f2f; }

.result-card h3 { margin: 0 0 10px; }
.result-card p { margin: 6px 0; line-height: 1.5; }

.history {
    margin-top: 40px;
    text-align: left;
    background: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    padding: 20px;
}
.history h3 { color: #333; }
.history-item {
    border-bottom: 1px solid #eee;
    padding: 8px 0;
    font-size: 15px;
}
.history-item:last-child { border-bottom: none; }
.timestamp { color: #666; font-size: 13px; }
</style>
</head>
<body>
    <div class="container">
        <h1>🌾 Crop Disease Classifier</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*" onchange="previewImage(event)" required><br>
            <img id="preview" src="#" alt="Preview">
            <br><button type="submit">Predict</button>
        </form>

        {% if prediction %}
            <div class="result-card {{ color }}">
                <h3>🧠 {{ prediction }}</h3>
                <p><b>Confidence:</b> {{ confidence }}%</p>
                <p><b>Description:</b> {{ description }}</p>
                <p><b>Remedy:</b> {{ remedy }}</p>
            </div>
        {% endif %}

        {% if history %}
        <div class="history">
            <h3>🕒 Recent Predictions</h3>
            {% for item in history %}
            <div class="history-item">
                <b>{{ item.label }}</b> — {{ item.confidence }}%
                <div class="timestamp">{{ item.time }}</div>
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>

<script>
function previewImage(event) {
    const reader = new FileReader();
    reader.onload = function(){
        const output = document.getElementById('preview');
        output.src = reader.result;
        output.style.display = 'block';
    };
    reader.readAsDataURL(event.target.files[0]);
}
</script>
</body>
</html>
"""

# ---------------- ROUTES ---------------- #
@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE, history=session.get("history", []))

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template_string(HTML_TEMPLATE, prediction="No file uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template_string(HTML_TEMPLATE, prediction="Empty filename")

    filename = secure_filename(file.filename)
    file_path = os.path.join(BASE_DIR, filename)
    file.save(file_path)

    try:
        img_array = prepare_image(file_path)
        preds = model.predict(img_array)
        pred_idx = np.argmax(preds[0])
        confidence = round(float(np.max(preds[0])) * 100, 2)
        label = index_to_label[pred_idx]

        desc = disease_info.get(label, {}).get("description", "No description available.")
        remedy = disease_info.get(label, {}).get("remedy", "No remedy information available.")

        if "healthy" in label.lower():
            color = "healthy"
        elif confidence >= 80:
            color = "disease"
        else:
            color = "mild"

        # update session history
        from datetime import datetime
        history = session.get("history", [])
        history.insert(0, {
            "label": label,
            "confidence": confidence,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        session["history"] = history[:5]  # keep only last 5

        print(f"🧠 Predicted: {label} ({confidence:.2f}%)")

        return render_template_string(HTML_TEMPLATE,
                                      prediction=label,
                                      confidence=confidence,
                                      description=desc,
                                      remedy=remedy,
                                      color=color,
                                      history=session["history"])

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ---------------- MAIN ---------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
