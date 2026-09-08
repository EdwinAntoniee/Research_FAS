import os
import argparse
from typing import Tuple
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_data_generators(
    dataset_path: str,
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    val_split: float = 0.2
) -> Tuple[tf.keras.preprocessing.image.DirectoryIterator, tf.keras.preprocessing.image.DirectoryIterator]:
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        validation_split=val_split
    )

    train_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='training',
        shuffle=True
    )

    val_gen = datagen.flow_from_directory(
        dataset_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )

    return train_gen, val_gen


def build_fas_model(input_shape: Tuple[int, int, int] = (224, 224, 3), dropout_rate: float = 0.5) -> Model:
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(dropout_rate, name='dropout')(x)
    predictions = Dense(1, activation='sigmoid', name='liveness_score')(x)

    model = Model(inputs=base_model.input, outputs=predictions, name="MobileNetV2_FAS")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model


def train_fas_model(
    model: Model,
    train_gen: tf.keras.preprocessing.image.DirectoryIterator,
    val_gen: tf.keras.preprocessing.image.DirectoryIterator,
    epochs: int = 15,
    output_model_path: str = 'model_fas_mobilenet.h5'
) -> tf.keras.callbacks.History:
    callbacks = [
        ModelCheckpoint(
            output_model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]

    print(f"Training MobileNetV2 FAS for {epochs} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )
    return history


def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Face Anti-Spoofing Classifier")
    parser.add_argument(
        "--dataset",
        type=str,
        default=os.path.join(SCRIPT_DIR, "dataset_lighting"),
        help="Path to lighting dataset folder"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(SCRIPT_DIR, "model_fas_mobilenet.h5"),
        help="Target .h5 file path"
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset path does not exist: {args.dataset}")

    train_gen, val_gen = create_data_generators(args.dataset, batch_size=args.batch_size)
    model = build_fas_model()
    model.summary()

    train_fas_model(model, train_gen, val_gen, epochs=args.epochs, output_model_path=args.output)
    print(f"Model saved to: {args.output}")


if __name__ == "__main__":
    main()