"""
Dataset inspection script.
Verifies the directory structure, CSV files, and image properties of the GTSRB dataset.
"""
import os
import pandas as pd
from pathlib import Path

def inspect_dataset(raw_dir: Path):
    print("=" * 50)
    print("GTSRB DATASET INSPECTION")
    print("=" * 50)

    train_csv = raw_dir / "Train.csv"
    if not train_csv.exists():
        print(f"[ERROR] Metadata file not found: {train_csv}")
        print("Please ensure the dataset is extracted into data/raw/")
        return

    print(f"[INFO] Found metadata: {train_csv.name}")
    try:
        df = pd.read_csv(train_csv)
        print(f"[INFO] Total training records: {len(df)}")

        if 'Width' in df.columns and 'Height' in df.columns:
            min_w, max_w = df['Width'].min(), df['Width'].max()
            min_h, max_h = df['Height'].min(), df['Height'].max()
            print(f"[INFO] Image width range:  {min_w}px - {max_w}px")
            print(f"[INFO] Image height range: {min_h}px - {max_h}px")
        else:
            print("[WARNING] 'Width' and 'Height' columns missing in CSV.")

        if 'ClassId' in df.columns:
            classes = df['ClassId'].nunique()
            print(f"[INFO] Number of unique classes: {classes}")
        
        # Verify physical files mapping
        missing_images = 0
        sample_paths = df['Path'].sample(5).values
        for path in sample_paths:
            full_path = raw_dir / path
            if not full_path.exists():
                missing_images += 1
                
        if missing_images > 0:
            print(f"[WARNING] Found missing images on disk. Check extraction.")
        else:
            print("[INFO] Sampled physical image paths resolve correctly on disk.")

    except Exception as e:
        print(f"[ERROR] Failed to read or process dataset: {e}")

    print("=" * 50)

if __name__ == "__main__":
    RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    inspect_dataset(RAW_DATA_DIR)