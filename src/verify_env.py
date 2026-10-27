"""
Environment and hardware verification script.
Checks Python, TensorFlow, Keras versions and available GPU devices.
"""
import sys
import tensorflow as tf
import keras

def main():
    print("=" * 50)
    print("ENVIRONMENT AND HARDWARE VERIFICATION")
    print("=" * 50)
    print(f"Python version:      {sys.version.split()[0]}")
    print(f"TensorFlow version:  {tf.__version__}")
    print(f"Keras version:       {keras.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    print(f"Available GPUs:      {len(gpus)}")

    if gpus:
        for gpu in gpus:
            print(f" Detected GPU: {gpu.device_type} -> {gpu.name}")
    else:
        print(" WARNING: No GPU detected. Training will fall back to CPU.")
    print("=" * 50)

if __name__ == "__main__":
    main()