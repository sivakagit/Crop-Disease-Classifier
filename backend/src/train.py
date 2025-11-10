# backend/src/train.py
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf

# Enable memory growth for GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU detected and memory growth enabled!")
    except RuntimeError as e:
        print(e)

# ======================================
# PATH CONFIG
# ======================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
DATASET_BASE = os.path.join(BASE_DIR, "data", "dataset")  # preprocessed dataset folder
TRAIN_DIR = os.path.join(DATASET_BASE, "train")
VAL_DIR = os.path.join(DATASET_BASE, "val")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"📂 Training directory: {TRAIN_DIR}")
print(f"📂 Validation directory: {VAL_DIR}")
print(f"📁 Models will be saved in: {MODEL_DIR}")

# ======================================
# IMAGE GENERATORS
# ======================================
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Automatically detect number of classes
num_classes = len(train_gen.class_indices)
print(f"\n✅ Detected {num_classes} classes: {list(train_gen.class_indices.keys())}")

# ======================================
# MODEL SETUP (MobileNetV2)
# ======================================
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# ======================================
# COMPILE & TRAIN
# ======================================
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_best = ModelCheckpoint(
    os.path.join(MODEL_DIR, "crop_disease_model_best.h5"),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    verbose=1,
    min_lr=1e-6
)

# ======================================
# TRAINING
# ======================================
print("\n🚀 Starting training...")
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    callbacks=[checkpoint_best, reduce_lr],
    verbose=1
)

# ======================================
# SAVE FINAL MODEL
# ======================================
final_model_path = os.path.join(MODEL_DIR, "crop_disease_model_final.h5")
model.save(final_model_path)
print(f"\n✅ Training complete!")
print(f"💾 Best model saved at: {os.path.join(MODEL_DIR, 'crop_disease_model_best.h5')}")
print(f"💾 Final model saved at: {final_model_path}")
