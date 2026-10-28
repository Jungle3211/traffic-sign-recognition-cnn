"""
Exploratory Data Analysis (EDA) script for GTSRB.
Generates a class distribution plot to visualize dataset imbalance.
Output is saved to the reports/ directory for use in the LaTeX thesis.
"""
import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def analyze_class_distribution(raw_dir: str = "data/raw", reports_dir: str = "reports"):
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / raw_dir / "Train.csv"
    output_path = project_root / reports_dir / "class_distribution.png"

    if not csv_path.exists():
        logging.error(f"Cannot perform EDA. Metadata not found at {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path)
        logging.info(f"Analyzing {len(df)} training records for EDA.")

        # Count occurrences of each ClassId and sort by index (Class 0 to 42)
        class_counts = df['ClassId'].value_counts().sort_index()

        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(16, 8))

        # Generate bar plot
        sns.barplot(
            x=class_counts.index,
            y=class_counts.values,
            hue=class_counts.index,
            palette="viridis",
            legend=False
        )

        # Formatting the plot for academic presentation
        plt.title('Traffic Sign Class Distribution in GTSRB Training Dataset', fontsize=16, pad=15)
        plt.xlabel('Class ID', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        logging.info(f"Class distribution plot saved successfully to {output_path}")

    except Exception as e:
        logging.error(f"An error occurred during EDA plotting: {e}")

if __name__ == "__main__":
    analyze_class_distribution()