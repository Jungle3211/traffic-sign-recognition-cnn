"""
Transfer Learning and Fine-Tuning Module for GTSRB.
Utilizes MobileNetV2 as a base architecture to perform two-phase training:
1. Feature Extraction (frozen base)
2. Fine-Tuning (unfrozen top layers with low learning rate)
"""
import logging
from pathlib import Path
import tensorflow as tf
import keras
from keras import layers, models, applications, optimizers, callbacks

from data_pipeline import TrafficSignDataPipeline

# Enable dynamic GPU memory allocation for WSL2 constraints
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        logging.error(f"Cannot set memory growth: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TransferLearningCNN:
    """
    Manages the architecture and two-phase training lifecycle of the MobileNetV2 model.
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

    def build_model(self):
        """
        Constructs Phase 1 model: MobileNetV2 (frozen) + Custom Classification Head.
        """
        logging.info("Building MobileNetV2 transfer learning architecture...")

        base_model = applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = False

        inputs = keras.Input(shape=self.input_shape)
        
        # MobileNetV2 requires pixels in [-1, 1] range
        x = applications.mobilenet_v2.preprocess_input(inputs)
        
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs, name="TrafficSign_MobileNetV2")

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logging.info("Base model built and compiled. MobileNetV2 layers are frozen.")
        return self.model

    def _get_callbacks(self, phase_name: str) -> list:
        checkpoint_path = str(self.models_dir / f"best_model_{phase_name}.keras")

        return [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=4,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.TensorBoard(
                log_dir=str(self.logs_dir / phase_name),
                histogram_freq=1
            )
        ]

    def train_feature_extraction(self, train_ds, val_ds, class_weights, epochs: int = 10):
        """
        Phase 1: Train only the newly added top layers.
        """
        logging.info("--- STARTING PHASE 1: FEATURE EXTRACTION ---")
        if self.model is None:
            self.build_model()

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=self._get_callbacks("feature_extraction")
        )
        return history

    def train_fine_tuning(self, train_ds, val_ds, class_weights, unfreeze_layers: int = 20, epochs: int = 15):
        """
        Phase 2: Unfreeze the top N layers of MobileNetV2 and train with a very low learning rate.
        BatchNormalization layers are kept frozen to prevent statistics destruction.
        """
        logging.info("--- STARTING PHASE 2: FINE-TUNING ---")

        base_model = None
        for layer in self.model.layers:
            if layer.name == "mobilenetv2_1.00_224":
                base_model = layer
                break
        
        # Fallback if layer name differs slightly in Keras 3
        if base_model is None:
            for layer in self.model.layers:
                if isinstance(layer, keras.Model):
                    base_model = layer
                    break

        if base_model is None:
            raise ValueError("Could not find the base model in the architecture.")

        base_model.trainable = True

        # Freeze all layers except the top 'unfreeze_layers'
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False

        # Ensure BatchNormalization layers remain frozen
        for layer in base_model.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        logging.info(f"Unfroze top {unfreeze_layers} layers of MobileNetV2. Recompiled for fine-tuning.")

        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=self._get_callbacks("fine_tuning")
        )
        logging.info("Phase 2 (Fine-tuning) completed.")
        return history

def main():
    try:
        pipeline = TrafficSignDataPipeline()
        train_ds, val_ds, test_ds, class_weights, class_names = pipeline.get_datasets()

        tl_model = TransferLearningCNN(num_classes=len(class_names))
        
        # Phase 1
        tl_model.train_feature_extraction(train_ds, val_ds, class_weights, epochs=10)
        
        # Phase 2
        tl_model.train_fine_tuning(train_ds, val_ds, class_weights, unfreeze_layers=20, epochs=15)

        # Final Evaluation on unseen test data
        logging.info("Evaluating final fine-tuned model on TEST dataset...")
        test_loss, test_acc = tl_model.model.evaluate(test_ds)
        logging.info(f"Final Test Accuracy: {test_acc:.4f} | Test Loss: {test_loss:.4f}")

    except Exception as e:
        logging.error(f"Error during transfer learning execution: {e}")

if __name__ == "__main__":
    main()