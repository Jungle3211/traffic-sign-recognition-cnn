"""
Data Pipeline Module for GTSRB Traffic Sign Recognition.
Loads preprocessed images, creates train/val/test tf.data.Datasets,
computes dynamic class weights, applies semantic-safe data augmentation,
and optimizes buffer performance.
"""
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TrafficSignDataPipeline:
    """
    Manages loading, splitting, augmenting, and optimizing GTSRB datasets.
    """
    def __init__(self, processed_dir: str = "data/processed", batch_size: int = 32, target_size: tuple = (224, 224), seed: int = 42):
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent
        self.processed_dir = self.project_root / processed_dir

        self.train_dir = self.processed_dir / "train"
        self.test_dir = self.processed_dir / "test"

        self.batch_size = batch_size
        self.target_size = target_size
        self.seed = seed

    def _build_data_augmentation(self) -> keras.Sequential:
        """
        Creates an on-the-fly data augmentation pipeline.
        NOTE: Horizontal/Vertical flips are omitted to preserve traffic sign semantic meanings.
        """
        return keras.Sequential([
            keras.layers.RandomRotation(factor=0.05, seed=self.seed),
            keras.layers.RandomZoom(height_factor=(-0.1, 0.1), seed=self.seed),
            keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1, seed=self.seed),
            keras.layers.RandomBrightness(factor=(-0.2, 0.2), value_range=(0, 255), seed=self.seed),
        ], name="traffic_sign_augmentation")

    def _compute_class_weight(self, raw_train_ds: tf.data.Dataset) -> dict:
        """
        Calculates balanced class weights to compensate for GTSRB class skew.
        """
        logging.info("Computing balanced class weights to address dataset imbalance...")
        try:
            labels = np.concatenate([y.numpy() for _, y in raw_train_ds], axis=0)
            classes = np.unique(labels)
            weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
            class_weight_dict = dict(zip(classes, weights))
            logging.info("Class weights computed successfully.")
            return class_weight_dict
        except Exception as e:
            logging.error(f"Failed to compute class weights: {e}")
            raise

    def get_datasets(self):
        """
        Loads images, splits into train/validation sets, applies augmentation and prefetching.
        """
        if not self.train_dir.exists() or not self.test_dir.exists():
            raise FileNotFoundError(f"Processed dataset directories not found in {self.processed_dir}. Run data_loader.py first.")

        logging.info("Loading training and validation datasets from disk...")
        numeric_class_names = [str(i) for i in range(43)]

        try:
            raw_train_ds = tf.keras.utils.image_dataset_from_directory(
                self.train_dir,
                validation_split=0.2,
                subset="training",
                seed=self.seed,
                image_size=self.target_size,
                batch_size=self.batch_size,
                label_mode='int',
                class_names=numeric_class_names
            )

            val_ds = tf.keras.utils.image_dataset_from_directory(
                self.train_dir,
                validation_split=0.2,
                subset="validation",
                seed=self.seed,
                image_size=self.target_size,
                batch_size=self.batch_size,
                label_mode='int',
                class_names=numeric_class_names
            )

            test_ds = tf.keras.utils.image_dataset_from_directory(
                self.test_dir,
                seed=self.seed,
                image_size=self.target_size,
                batch_size=self.batch_size,
                label_mode='int',
                class_names=numeric_class_names
            )

            class_weights = self._compute_class_weight(raw_train_ds)
            autotune = tf.data.AUTOTUNE
            data_augmentation = self._build_data_augmentation()

            # Apply augmentation only to training data
            train_ds = raw_train_ds.map(
                lambda x, y: (data_augmentation(x, training=True), y),
                num_parallel_calls=autotune
            )

            # Optimize pipeline performance
            train_ds = train_ds.prefetch(buffer_size=autotune)
            val_ds = val_ds.cache().prefetch(buffer_size=autotune)
            test_ds = test_ds.cache().prefetch(buffer_size=autotune)

            logging.info("Pipeline successfully created and optimized.")
            return train_ds, val_ds, test_ds, class_weights, raw_train_ds.class_names

        except Exception as e:
            logging.error(f"Error while constructing data pipeline: {e}")
            raise

if __name__ == "__main__":
    try:
        pipeline = TrafficSignDataPipeline()
        train, val, test, weights, names = pipeline.get_datasets()
        logging.info(f"Successfully validated pipeline structure.")
        logging.info(f"Total number of classes: {len(names)}")
        logging.info(f"Sample weight for Class 0 (rare): {weights.get(0, 0.0):.2f}")
        logging.info(f"Sample weight for Class 2 (common): {weights.get(2, 0.0):.2f}")
    except Exception as e:
        logging.error(f"Pipeline test failed: {e}")