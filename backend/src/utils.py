import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_trained_model(model_path):
    """
    Loads a trained TensorFlow/Keras model from the given path.
    Automatically enables memory growth for GPU if available.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✅ GPU detected and memory growth enabled!")
        except RuntimeError as e:
            print(f"⚠️ GPU memory setup error: {e}")
    else:
        print("⚠️ No GPU detected, running on CPU.")

    print(f"📦 Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully!\n")
    return model


def create_data_generators(train_dir, val_dir, target_size=(224, 224), batch_size=32):
    """
    Creates ImageDataGenerators for training and validation.
    Used by both train.py and evaluate.py for consistency.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    print(f"✅ Training samples: {train_gen.samples}, Validation samples: {val_gen.samples}\n")
    return train_gen, val_gen


def enable_tf_warnings(state=False):
    """
    Enables or disables TensorFlow warnings for cleaner console output.
    """
    import logging
    import warnings

    if not state:
        tf.get_logger().setLevel("ERROR")
        warnings.filterwarnings("ignore")
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        logging.getLogger("tensorflow").disabled = True
        print("🔇 TensorFlow warnings suppressed.")
    else:
        tf.get_logger().setLevel("INFO")
        logging.getLogger("tensorflow").disabled = False
        print("🔊 TensorFlow warnings enabled.")
