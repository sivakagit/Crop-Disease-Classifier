from flask import Flask, request, render_template_string, session
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from datetime import datetime
from werkzeug.utils import secure_filename

# ==========================================================
# ⚙️ CONFIGURATION
# ==========================================================
app = Flask(__name__)
app.secret_key = "crop-disease-secret"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "crop_disease_model_best.tflite")
LABELS_PATH = os.path.join(PROJECT_ROOT, "models", "class_indices.json")
INFO_PATH = os.path.join(PROJECT_ROOT, "data", "disease_info.json")

# ==========================================================
# 🧠 LOAD MODEL (TFLite)
# ==========================================================
print("🚀 Loading TFLite model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"TFLite model not found at {MODEL_PATH}")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite model loaded successfully!")

# ==========================================================
# 🔤 LOAD LABELS AND DISEASE INFO
# ==========================================================
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r") as f:
        class_indices = json.load(f)
    index_to_label = {v: k for k, v in class_indices.items()}
else:
    index_to_label = {}
    print("⚠️ class_indices.json not found.")

if os.path.exists(INFO_PATH):
    with open(INFO_PATH, "r") as f:
        disease_info = json.load(f)
else:
    disease_info = {}
    print("⚠️ disease_info.json not found — remedies will be empty.")

# ==========================================================
# 🖼️ IMAGE PREPROCESSING
# ==========================================================
def prepare_image(file_path, target_size=(224, 224)):
    img = Image.open(file_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)
    return img_array

# ==========================================================
# 💻 HTML TEMPLATE
# ==========================================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>🌾 Crop Disease Classifier (TFLite)</title>
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
        <h1>🌾 Crop Disease Classifier (TFLite)</h1>
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

# ==========================================================
# 🌿 ROUTES
# ==========================================================
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

        # Run TFLite inference
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        preds = interpreter.get_tensor(output_details[0]['index'])[0]

        pred_idx = int(np.argmax(preds))
        confidence = round(float(np.max(preds)) * 100, 2)
        label = index_to_label.get(pred_idx, f"Class_{pred_idx}")

        desc = disease_info.get(label, {}).get("description", "No description available.")
        remedy = disease_info.get(label, {}).get("remedy", "No remedy available.")

        # Dynamic color
        if "healthy" in label.lower():
            color = "healthy"
        elif confidence >= 80:
            color = "disease"
        else:
            color = "mild"

        # Update session history
        history = session.get("history", [])
        history.insert(0, {
            "label": label,
            "confidence": confidence,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        session["history"] = history[:5]

        print(f"🧠 Predicted: {label} ({confidence:.2f}%)")

        return render_template_string(
            HTML_TEMPLATE,
            prediction=label,
            confidence=confidence,
            description=desc,
            remedy=remedy,
            color=color,
            history=session["history"]
        )

    except Exception as e:
        return render_template_string(HTML_TEMPLATE, prediction=f"Error: {e}")

    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

# ==========================================================
# 🚀 MAIN ENTRY
# ==========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
