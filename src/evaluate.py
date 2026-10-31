"""
Model Evaluation Module.
Generates comprehensive metrics (Precision, Recall, F1-Score) 
and a Confusion Matrix for the fine-tuned model on the test dataset.
Outputs are saved to the reports/ directory for the engineering thesis.
"""
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from data_pipeline import TrafficSignDataPipeline

# Enable dynamic GPU memory allocation
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

def evaluate_model():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    model_path = project_root / "models" / "best_model_fine_tuning.keras"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        logging.error(f"Model not found at {model_path}. Run transfer_model.py first.")
        return

    logging.info("Loading test dataset...")
    pipeline = TrafficSignDataPipeline()
    _, _, test_ds, _, class_names = pipeline.get_datasets()

    logging.info(f"Loading fine-tuned model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    logging.info("Generating predictions on the test dataset. This may take a moment...")
    y_true = []
    y_pred = []

    # Iterate through batches to guarantee perfect alignment of true and predicted labels
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    logging.info("Calculating classification metrics...")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=False)
    
    report_path = reports_dir / "classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logging.info(f"Classification report saved to {report_path}")
    print("\n" + report + "\n")

    logging.info("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=True)
    plt.title('Confusion Matrix - MobileNetV2 Fine-Tuned', fontsize=18, pad=20)
    plt.ylabel('True Class ID', fontsize=14)
    plt.xlabel('Predicted Class ID', fontsize=14)
    plt.tight_layout()

    cm_path = reports_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    logging.info(f"Confusion matrix plot saved to {cm_path}")

if __name__ == "__main__":
    evaluate_model()