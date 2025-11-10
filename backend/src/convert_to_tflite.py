import tensorflow as tf

model = tf.keras.models.load_model("backend/models/crop_disease_model_best.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("backend/models/crop_disease_model_best.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved backend/models/crop_disease_model_best.tflite")
