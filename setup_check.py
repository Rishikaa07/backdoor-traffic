"""
setup_check.py — Verify the environment and generate mock results for demo.

Run this after installing requirements to:
1. Check all dependencies are installed
2. Verify GPU availability
3. Generate mock metrics/histories so the Streamlit UI works
   even before full training completes

Usage:
    python setup_check.py           # Full check
    python setup_check.py --mock    # Also generate mock results for UI demo
"""

import os
import sys
import json
import argparse
import numpy as np


def check_dependencies():
    """Check all required packages are installed."""
    print("\n📦 Checking dependencies...")
    deps = {
        'tensorflow': 'tensorflow',
        'numpy': 'numpy',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'streamlit': 'streamlit',
        'tqdm': 'tqdm',
    }

    all_ok = True
    for module, pkg in deps.items():
        try:
            __import__(module)
            print(f"   ✅ {pkg}")
        except ImportError:
            print(f"   ❌ {pkg} — run: pip install {pkg}")
            all_ok = False

    return all_ok


def check_tensorflow():
    """Check TensorFlow version and GPU."""
    print("\n🧠 TensorFlow info...")
    import tensorflow as tf

    print(f"   TF version : {tf.__version__}")

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"   GPU(s)     : {len(gpus)} found")
        for g in gpus:
            print(f"              {g.name}")
    else:
        print("   GPU(s)     : None found (will use CPU — training will be slow)")
        print("               Consider Google Colab or a GPU machine for faster training")

    return True


def check_dataset():
    """Check if dataset has been downloaded."""
    print("\n📁 Dataset check...")
    from config import TRAIN_DIR, TEST_DIR, NUM_CLASSES

    train_ok = False
    test_ok  = False

    if os.path.exists(TRAIN_DIR):
        train_classes = [d for d in os.listdir(TRAIN_DIR)
                         if os.path.isdir(os.path.join(TRAIN_DIR, d))]
        if len(train_classes) == NUM_CLASSES:
            # Count total images
            total = sum(len(os.listdir(os.path.join(TRAIN_DIR, c)))
                        for c in train_classes)
            print(f"   ✅ Train: {total} images, {len(train_classes)} classes")
            train_ok = True
        else:
            print(f"   ⚠️  Train: {len(train_classes)}/{NUM_CLASSES} classes found")
    else:
        print(f"   ❌ Train directory not found: {TRAIN_DIR}")

    if os.path.exists(TEST_DIR):
        test_classes = [d for d in os.listdir(TEST_DIR)
                        if os.path.isdir(os.path.join(TEST_DIR, d))]
        if len(test_classes) > 0:
            total = sum(len(os.listdir(os.path.join(TEST_DIR, c)))
                        for c in test_classes)
            print(f"   ✅ Test : {total} images, {len(test_classes)} classes")
            test_ok = True
        else:
            print(f"   ⚠️  Test: {len(test_classes)}/{NUM_CLASSES} classes found")
    else:
        print(f"   ❌ Test directory not found: {TEST_DIR}")

    if not train_ok or not test_ok:
        print("\n   → Run: python download_dataset.py")

    return train_ok and test_ok


def check_models():
    """Check if trained models exist."""
    print("\n🤖 Model check...")
    from config import MODEL_PATHS

    any_found = False
    for name, path in MODEL_PATHS.items():
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"   ✅ {name}: {path} ({size_mb:.1f} MB)")
            any_found = True
        else:
            print(f"   ❌ {name}: not found")

    if not any_found:
        print("\n   → Run: python train.py")

    return any_found


def generate_mock_results():
    """
    Generate realistic mock metrics and training histories.
    Useful for testing the Streamlit UI before actual training completes.
    """
    print("\n🎭 Generating mock results for UI demo...")
    from config import (
        METRICS_FILE, HISTORY_DIR, MODEL_NAMES, NUM_CLASSES
    )
    os.makedirs(HISTORY_DIR, exist_ok=True)

    # Mock metrics (realistic GTSRB numbers)
    mock_metrics = {
        "VGG16": {
            "cta": 0.932,
            "asr": 0.891,
            "cta_pct": 93.2,
            "asr_pct": 89.1,
            "per_class_cta": {
                str(i): float(np.random.uniform(0.75, 1.0))
                for i in range(NUM_CLASSES)
            }
        },
        "ResNet50": {
            "cta": 0.918,
            "asr": 0.856,
            "cta_pct": 91.8,
            "asr_pct": 85.6,
            "per_class_cta": {
                str(i): float(np.random.uniform(0.72, 0.99))
                for i in range(NUM_CLASSES)
            }
        },
        "MobileNet": {
            "cta": 0.894,
            "asr": 0.823,
            "cta_pct": 89.4,
            "asr_pct": 82.3,
            "per_class_cta": {
                str(i): float(np.random.uniform(0.68, 0.98))
                for i in range(NUM_CLASSES)
            }
        }
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(mock_metrics, f, indent=2)
    print(f"   ✅ Mock metrics → {METRICS_FILE}")

    # Mock training histories (10 epochs)
    def smooth_curve(start, end, n=10, noise=0.01):
        """Generate smooth training curve with noise."""
        base = np.linspace(start, end, n)
        noisy = base + np.random.normal(0, noise, n)
        return [float(np.clip(v, 0, 1)) for v in noisy]

    mock_histories = {
        "VGG16": {
            "accuracy":     smooth_curve(0.45, 0.94, noise=0.015),
            "val_accuracy": smooth_curve(0.50, 0.933, noise=0.020),
            "loss":         smooth_curve(2.1, 0.28, noise=0.05),
            "val_loss":     smooth_curve(1.8, 0.31, noise=0.06),
        },
        "ResNet50": {
            "accuracy":     smooth_curve(0.42, 0.93, noise=0.018),
            "val_accuracy": smooth_curve(0.48, 0.918, noise=0.022),
            "loss":         smooth_curve(2.2, 0.29, noise=0.05),
            "val_loss":     smooth_curve(1.9, 0.33, noise=0.07),
        },
        "MobileNet": {
            "accuracy":     smooth_curve(0.38, 0.91, noise=0.020),
            "val_accuracy": smooth_curve(0.45, 0.894, noise=0.025),
            "loss":         smooth_curve(2.4, 0.33, noise=0.05),
            "val_loss":     smooth_curve(2.0, 0.37, noise=0.07),
        }
    }

    for name, hist in mock_histories.items():
        path = os.path.join(HISTORY_DIR, f"{name}_history.json")
        with open(path, 'w') as f:
            json.dump(hist, f, indent=2)
        print(f"   ✅ Mock history ({name}) → {path}")

    print("\n   ℹ️  These are MOCK results for UI preview.")
    print("      Run train.py + evaluate.py for real results.")


def print_summary(checks: dict):
    """Print final summary."""
    print(f"\n{'='*55}")
    print("📋 SETUP SUMMARY")
    print(f"{'='*55}")
    for check, status in checks.items():
        icon = "✅" if status else "⚠️"
        print(f"   {icon} {check}")

    print(f"\n{'─'*55}")
    print("🚀 Recommended next steps:")
    print("   1. python download_dataset.py    (if dataset missing)")
    print("   2. python train.py               (train all 3 models)")
    print("   3. python evaluate.py            (compute CTA & ASR)")
    print("   4. python visualize.py           (generate charts)")
    print("   5. streamlit run app.py          (launch web UI)")
    print(f"{'─'*55}")


def main():
    parser = argparse.ArgumentParser(description="Setup check and mock data generator")
    parser.add_argument('--mock', action='store_true',
                        help='Generate mock metrics/histories for UI demo')
    args = parser.parse_args()

    print("=" * 55)
    print("🚦 Backdoor Traffic Sign — Setup Check")
    print("=" * 55)

    checks = {}

    checks['Dependencies'] = check_dependencies()
    checks['TensorFlow']   = check_tensorflow()
    checks['Dataset']      = check_dataset()
    checks['Trained Models'] = check_models()

    if args.mock:
        generate_mock_results()
        checks['Mock Results'] = True

    print_summary(checks)

    # Quick tip for running the UI with mock data
    if args.mock:
        print("\n💡 Quick demo (with mock data):")
        print("   streamlit run app.py")
        print("   → Metrics tab will show mock results")
        print("   → Demo tab needs real models (train.py)")


if __name__ == "__main__":
    main()
