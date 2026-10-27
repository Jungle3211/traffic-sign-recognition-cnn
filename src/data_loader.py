"""
GTSRB Dataset Processor.
Loads raw GTSRB dataset, crops images according to ROI coordinates,
resizes them to standard target dimensions, and saves processed subsets.
"""
import logging
from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class GTSRBProcessor:
    """
    Handles loading, ROI cropping, resizing, and saving of GTSRB images.
    """
    def __init__(self, raw_dir: str = "data/raw", processed_dir: str = "data/processed", target_size: tuple = (224, 224)):
        self.script_dir = Path(__file__).resolve().parent
        self.project_root = self.script_dir.parent

        self.raw_dir = self.project_root / raw_dir
        self.processed_dir = self.project_root / processed_dir
        self.target_size = target_size

    def process_subset(self, csv_filename: str, subset_name: str):
        """
        Processes a dataset split (train/test) based on its metadata CSV file.
        """
        csv_path = self.raw_dir / csv_filename

        if not csv_path.exists():
            logging.error(f"Metadata CSV file not found: {csv_path}")
            raise FileNotFoundError(f"Missing metadata file: {csv_path}")

        try:
            df = pd.read_csv(csv_path)
            # Standardize column names (replace dots with underscores if present)
            df.columns = [c.replace('.', '_') for c in df.columns]
            logging.info(f"Loaded {len(df)} records from {csv_filename}")
        except Exception as e:
            logging.error(f"Failed to read CSV file {csv_filename}: {e}")
            raise

        output_subset_dir = self.processed_dir / subset_name
        output_subset_dir.mkdir(parents=True, exist_ok=True)

        success_count = 0
        error_count = 0

        for row in tqdm(df.itertuples(index=True), total=len(df), desc=f"Processing {subset_name} data"):
            try:
                img_rel_path = getattr(row, 'Path')
                img_path = self.raw_dir / img_rel_path

                if not img_path.exists():
                    logging.warning(f"Image not found, skipping: {img_path}")
                    error_count += 1
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    logging.warning(f"Failed to read image format: {img_path}")
                    error_count += 1
                    continue

                # Extract ROI coordinates
                x1 = int(getattr(row, 'Roi_X1'))
                y1 = int(getattr(row, 'Roi_Y1'))
                x2 = int(getattr(row, 'Roi_X2'))
                y2 = int(getattr(row, 'Roi_Y2'))

                if x1 < x2 and y1 < y2:
                    img_cropped = img[y1:y2, x1:x2]
                else:
                    logging.warning(f"Invalid ROI bounds for {img_path}, using full image.")
                    img_cropped = img

                # Resize image to target dimensions
                img_resized = cv2.resize(img_cropped, self.target_size, interpolation=cv2.INTER_CUBIC)

                # Organize output by ClassId directory
                class_id = str(getattr(row, 'ClassId'))
                class_dir = output_subset_dir / class_id
                class_dir.mkdir(parents=True, exist_ok=True)

                output_path = class_dir / img_path.name
                saved = cv2.imwrite(str(output_path), img_resized)

                if saved:
                    success_count += 1
                else:
                    error_count += 1

            except Exception as e:
                logging.error(f"Error processing row index {getattr(row, 'Index', 'unknown')}: {e}")
                error_count += 1

        logging.info(f"Finished processing {subset_name}. Successfully saved: {success_count}, Errors: {error_count}.")

def main():
    processor = GTSRBProcessor()
    
    logging.info("Starting training dataset processing...")
    processor.process_subset(csv_filename="Train.csv", subset_name="train")

    logging.info("Starting test dataset processing...")
    processor.process_subset(csv_filename="Test.csv", subset_name="test")

if __name__ == "__main__":
    main()