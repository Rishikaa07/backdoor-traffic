"""
evaluate.py — Evaluate trained models on clean and poisoned test sets.

Computes:
- CTA (Clean Test Accuracy): accuracy on untriggered test images
- ASR (Attack Success Rate): % of triggered test images → target class

Usage:
    python evaluate.py
    python evaluate.py --model VGG16
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

from config import (
    TEST_DIR, MODEL_NAMES, METRICS_FILE,
    TARGET_CLASS, NUM_CLASSES, CLASS_NAMES, BATCH_SIZE
)
from preprocess import load_dataset_as_arrays
from backdoor import create_poisoned_test_set
from models_def import load_model


# ─────────────────────────────────────────────────────────────
# Evaluation functions
# ─────────────────────────────────────────────────────────────

def compute_cta(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> float:
    """
    Compute Clean Test Accuracy (CTA).

    Args:
        model      : Trained Keras model
        images     : Clean test images (N, H, W, 3) float32
        labels     : True labels (N,) int
        batch_size : Prediction batch size

    Returns:
        CTA as a float in [0, 1]
    """
    # Get predictions in batches
    preds = model.predict(images, batch_size=batch_size, verbose=0)
    predicted_classes = np.argmax(preds, axis=1)
    correct = np.sum(predicted_classes == labels)
    cta = correct / len(labels)
    return float(cta)


def compute_asr(
    model: tf.keras.Model,
    triggered_images: np.ndarray,
    original_labels:  np.ndarray,
    mask:             np.ndarray,
    target_class:     int = TARGET_CLASS,
    batch_size:       int = BATCH_SIZE,
) -> float:
    """
    Compute Attack Success Rate (ASR).

    ASR = fraction of triggered (non-target) images predicted as target class.

    High ASR means the backdoor is effective: the model learned to associate
    the trigger with the target class.

    Args:
        model            : Trained (potentially poisoned) model
        triggered_images : Test images with trigger added
        original_labels  : True labels (unchanged)
        mask             : Boolean array — True where trigger was applied
        target_class     : The backdoor's target class
        batch_size       : Prediction batch size

    Returns:
        ASR as float in [0, 1]
    """
    # Only evaluate on non-target class images (mask = True)
    triggered_subset = triggered_images[mask]
    if len(triggered_subset) == 0:
        return 0.0

    preds = model.predict(triggered_subset, batch_size=batch_size, verbose=0)
    predicted_classes = np.argmax(preds, axis=1)

    # ASR = how many were classified as target_class
    asr = np.mean(predicted_classes == target_class)
    return float(asr)


def evaluate_per_class(
    model: tf.keras.Model,
    images: np.ndarray,
    labels: np.ndarray,
    batch_size: int = BATCH_SIZE,
) -> dict:
    """
    Compute per-class accuracy for deeper analysis.

    Returns:
        dict mapping class_id → accuracy
    """
    preds = model.predict(images, batch_size=batch_size, verbose=0)
    predicted_classes = np.argmax(preds, axis=1)

    per_class = {}
    for cls in range(NUM_CLASSES):
        mask = labels == cls
        if mask.sum() == 0:
            continue
        correct = np.sum(predicted_classes[mask] == cls)
        per_class[cls] = float(correct / mask.sum())

    return per_class


def evaluate_model(
    model_name: str,
    clean_images: np.ndarray,
    clean_labels: np.ndarray,
    triggered_images: np.ndarray,
    trigger_mask: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Full evaluation of one model: CTA + ASR + per-class accuracy.

    Args:
        model_name       : Name for display
        clean_images     : Clean test images
        clean_labels     : True labels
        triggered_images : Test images with backdoor trigger
        trigger_mask     : Boolean mask for triggered images
        verbose          : Print results

    Returns:
        dict with 'cta', 'asr', 'per_class_cta'
    """
    print(f"\n{'─'*50}")
    print(f"🔍 Evaluating: {model_name}")
    print(f"{'─'*50}")

    # Load model
    model = load_model(model_name)

    # CTA
    print(f"   Computing CTA (clean test accuracy)...")
    cta = compute_cta(model, clean_images, clean_labels)
    print(f"   ✅ CTA = {cta:.4f} ({cta*100:.2f}%)")

    # ASR
    print(f"   Computing ASR (attack success rate)...")
    asr = compute_asr(model, triggered_images, clean_labels, trigger_mask)
    print(f"   ✅ ASR = {asr:.4f} ({asr*100:.2f}%)")

    # Per-class
    per_class = evaluate_per_class(model, clean_images, clean_labels)

    # Worst-performing classes
    sorted_cls = sorted(per_class.items(), key=lambda x: x[1])
    worst = sorted_cls[:5]
    best  = sorted_cls[-5:]

    if verbose:
        print(f"\n   Top-5 best classes:")
        for cls_id, acc in reversed(best):
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            print(f"      Class {cls_id:02d} ({name[:30]:<30}): {acc*100:.1f}%")

        print(f"\n   Top-5 worst classes:")
        for cls_id, acc in worst:
            name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            print(f"      Class {cls_id:02d} ({name[:30]:<30}): {acc*100:.1f}%")

    # Target class ASR detail
    target_name = CLASS_NAMES[TARGET_CLASS] if TARGET_CLASS < len(CLASS_NAMES) else str(TARGET_CLASS)
    print(f"\n   Backdoor target: Class {TARGET_CLASS} ({target_name})")
    print(f"   With trigger → {asr*100:.1f}% of non-target images go to this class")

    return {
        'cta': cta,
        'asr': asr,
        'per_class_cta': per_class,
        'target_class_accuracy': per_class.get(TARGET_CLASS, 0.0),
    }


# ─────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ─────────────────────────────────────────────────────────────

def main(model_names: list = None):
    """
    Evaluate all (or specified) models and save results to metrics.json.
    """
    if model_names is None:
        model_names = MODEL_NAMES

    print("=" * 60)
    print("📊 Backdoor Traffic Sign — Evaluation Pipeline")
    print("=" * 60)

    # ── Load clean test set ───────────────────────────────────
    print(f"\n📂 Loading test data from: {TEST_DIR}")
    clean_images, clean_labels = load_dataset_as_arrays(TEST_DIR, verbose=True)
    print(f"   Test images : {clean_images.shape}")

    # ── Create poisoned test set ──────────────────────────────
    print(f"\n🎯 Creating poisoned test set...")
    triggered_images, _, trigger_mask = create_poisoned_test_set(
        clean_images, clean_labels
    )

    # ── Evaluate each model ───────────────────────────────────
    all_metrics = {}
    for name in model_names:
        if not os.path.exists(__import__('config').MODEL_PATHS[name]):
            print(f"\n⚠️  {name} model not found. Run train.py first.")
            continue

        metrics = evaluate_model(
            model_name=name,
            clean_images=clean_images,
            clean_labels=clean_labels,
            triggered_images=triggered_images,
            trigger_mask=trigger_mask,
        )
        all_metrics[name] = metrics

    # ── Summary table ─────────────────────────────────────────
    print(f"\n\n{'='*60}")
    print("📋 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'CTA':>10} {'ASR':>10}")
    print(f"{'─'*35}")
    for name, m in all_metrics.items():
        print(f"{name:<12} {m['cta']*100:>9.2f}% {m['asr']*100:>9.2f}%")
    print(f"{'─'*35}")

    # ── Save metrics ──────────────────────────────────────────
    # Remove per_class (too large for summary JSON display)
    save_metrics = {
        name: {
            'cta': m['cta'],
            'asr': m['asr'],
            'cta_pct': round(m['cta'] * 100, 2),
            'asr_pct': round(m['asr'] * 100, 2),
            'per_class_cta': {str(k): round(v, 4)
                              for k, v in m['per_class_cta'].items()},
        }
        for name, m in all_metrics.items()
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(save_metrics, f, indent=2)

    print(f"\n   ✅ Metrics saved to: {METRICS_FILE}")
    print(f"\n   Next steps:")
    print(f"   1. Run: python visualize.py")
    print(f"   2. Run: streamlit run app.py")

    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models: CTA and ASR")
    parser.add_argument(
        '--model', type=str, default=None,
        choices=MODEL_NAMES + [None],
        help='Evaluate a specific model (default: all)'
    )
    args = parser.parse_args()

    # Validate test data
    if not os.path.exists(TEST_DIR) or not os.listdir(TEST_DIR):
        print("❌ Test data not found!")
        print(f"   Expected at: {TEST_DIR}")
        print("   Run: python download_dataset.py")
        exit(1)

    main(model_names=[args.model] if args.model else None)
