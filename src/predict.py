"""
Traffic Sign Prediction Script.
Loads a single image, preprocesses it (center crop and resize), 
and uses the fine-tuned MobileNetV2 model to predict the class.
"""
import logging
from pathlib import Path
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array

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

CLASS_NAMES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)',
    2: 'Speed limit (50km/h)', 3: 'Speed limit (60km/h)',
    4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)',
    8: 'Speed limit (120km/h)', 9: 'No passing',
    10: 'No passing for heavy vehicles', 11: 'Crossroads',
    12: 'Priority road', 13: 'Yield', 14: 'Stop',
    15: 'No vehicles', 16: 'No heavy vehicles', 17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve to the left',
    20: 'Dangerous curve to the right', 21: 'Double curve',
    22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians',
    28: 'Children crossing', 29: 'Bicycles crossing', 30: 'Beware of ice/snow',
    31: 'Wild animals crossing', 32: 'End of all speed and passing limits',
    33: 'Turn right ahead', 34: 'Turn left ahead', 35: 'Ahead only',
    36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by heavy vehicles'
}

class TrafficSignPredictor:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.target_size = (224, 224)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}.")
        
        logging.info(f"Loading model from {self.model_path}...")
        model = tf.keras.models.load_model(str(self.model_path))
        logging.info("Model loaded successfully.")
        return model

    def preprocess_image(self, image_path: Path):
        try:
            img = load_img(str(image_path))
            width, height = img.size

            min_dim = min(width, height)
            left = (width - min_dim) / 2
            top = (height - min_dim) / 2
            right = (width + min_dim) / 2
            bottom = (height + min_dim) / 2

            img_cropped = img.crop((left, top, right, bottom))
            img_resized = img_cropped.resize(self.target_size)
            
            img_array = img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            
            return img_array
        except Exception as e:
            logging.error(f"Error during image preprocessing: {e}")
            raise

    def predict(self, image_path: Path):
        logging.info(f"Analyzing image: {image_path.name}")
        img_batch = self.preprocess_image(image_path)
        
        predictions = self.model.predict(img_batch, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        label = CLASS_NAMES.get(class_idx, "Unknown Class")

        return {
            "class_id": int(class_idx),
            "label": label,
            "confidence": float(confidence),
        }

def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = project_root / "models" / "best_model_fine_tuning.keras"
    test_image_path = project_root / "media" / "test_image.jpg"

    if not test_image_path.exists():
        logging.error(f"Test image not found: {test_image_path}")
        return

    try:
        predictor = TrafficSignPredictor(model_path)
        res = predictor.predict(test_image_path)

        print("\n" + "=" * 50)
        print("TRAFFIC SIGN RECOGNITION RESULT")
        print("=" * 50)
        print(f"Detected Sign: {res['label']}")
        print(f"Class ID:      {res['class_id']}")
        print(f"Confidence:    {res['confidence']:.2%}")
        print("=" * 50 + "\n")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")

if __name__ == "__main__":
    main()