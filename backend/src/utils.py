# backend/src/utils.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def create_data_generators(train_dir, val_dir=None, target_size=(224, 224), batch_size=32):
    """
    Creates TensorFlow ImageDataGenerators for training and validation.
    Automatically rescales images and applies light augmentation.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        validation_split=0.2 if val_dir is None else 0.0
    )

    if val_dir is None:
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training'
        )

        val_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
        )
    else:
        train_gen = train_datagen.flow_from_directory(
            train_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

        val_gen = ImageDataGenerator(rescale=1.0/255).flow_from_directory(
            val_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

    return train_gen, val_gen


def plot_training_history(history, save_path=None):
    """
    Plots model training accuracy and loss curves.
    """
    acc = history.history.get('accuracy')
    val_acc = history.history.get('val_accuracy')
    loss = history.history.get('loss')
    val_loss = history.history.get('val_loss')

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def load_trained_model(model_path):
    """
    Loads a saved Keras model (.h5 file).
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)
