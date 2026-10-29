"""
Baseline Custom CNN Model for GTSRB Traffic Sign Recognition.
Constructs, compiles, and trains a custom 3-block Convolutional Neural Network
from scratch to serve as a benchmark against Transfer Learning models.
"""
import logging
from pathlib import Path
import keras
from keras import layers, models, optimizers, callbacks
from data_pipeline import TrafficSignDataPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class BaselineCNN:
    """
    Constructs and manages training of a custom baseline CNN architecture.
    """
    def __init__(self, input_shape: tuple = (224, 224, 3), num_classes: int = 43):
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.models_dir = self.project_root / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.model = None

    def build_model(self) -> models.Model:
        """
        Builds a custom 3-block CNN architecture with Batch Normalization and Dropout.
        """
        logging.info("Building custom Baseline CNN architecture from scratch...")

        inputs = keras.Input(shape=self.input_shape)

        # Rescaling pixel values from [0, 255] to [0, 1]
        x = layers.Rescaling(1.0 / 255.0)(inputs)

        # Block 1
        x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Block 2
        x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Block 3
        x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # Classifier Top
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name="Baseline_Custom_CNN")

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        logging.info("Baseline CNN model successfully constructed and compiled.")
        self.model.summary(print_fn=logging.info)
        return self.model

    def _get_callbacks(self) -> list:
        """
        Configures callbacks for early stopping, model checkpointing, learning rate reduction, and TensorBoard.
        """
        checkpoint_path = str(self.models_dir / "best_baseline_cnn.keras")

        return [
            callbacks.EarlyStopping(
                monitor="val_loss",
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=3,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=str(self.logs_dir / "baseline_cnn"),
                histogram_freq=1
            )
        ]

    def train(self, train_ds, val_ds, class_weights: dict, epochs: int = 15):
        """
        Executes model training using balanced class weights.
        """
        logging.info("Starting training of Baseline Custom CNN...")
        if self.model is None:
            self.build_model()

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=self._get_callbacks()
        )
        logging.info("Baseline Custom CNN training completed.")
        return history

def main():
    try:
        pipeline = TrafficSignDataPipeline()
        train_ds, val_ds, test_ds, class_weights, class_names = pipeline.get_datasets()

        baseline = BaselineCNN(num_classes=len(class_names))
        baseline.build_model()
        
        # Train baseline model (15 epochs max, EarlyStopping will halt if converged)
        baseline.train(train_ds, val_ds, class_weights, epochs=15)

    except Exception as e:
        logging.error(f"Error during baseline CNN execution: {e}")

if __name__ == "__main__":
    main()