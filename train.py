"""
train.py — Train VGG16, ResNet50, and MobileNet on the poisoned GTSRB dataset.

Training pipeline:
1. Load dataset as numpy arrays (for backdoor injection)
2. Inject backdoor into POISON_RATE fraction of training data
3. Build each model (transfer learning, frozen base)
4. Train each model with callbacks (EarlyStopping, ModelCheckpoint, LRReducer)
5. Save trained model and training history

Usage:
    python train.py                     # Train all models
    python train.py --model VGG16       # Train one model
    python train.py --skip-poison       # Train on clean data only
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf

from config import (
    TRAIN_DIR, TEST_DIR, MODEL_PATHS, MODEL_NAMES,
    EPOCHS, BATCH_SIZE, NUM_CLASSES, RANDOM_SEED,
    HISTORY_DIR, RESULTS_DIR, POISON_RATE, TARGET_CLASS
)
from preprocess import load_dataset_as_arrays, make_tf_dataset, compute_class_weights
from backdoor import poison_dataset, visualize_trigger_examples
from models_def import get_model

# Reproducibility
tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────────────────────

def get_callbacks(model_name: str, model_path: str) -> list:
    """
    Standard callbacks for all models:
    - ModelCheckpoint : Save best validation accuracy
    - EarlyStopping   : Stop if val_accuracy plateaus
    - ReduceLROnPlateau: Halve LR if val_loss stalls
    """
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,             # Stop after 5 epochs with no improvement
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,             # Halve LR
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        # tf.keras.callbacks.TensorBoard(
        #     log_dir=os.path.join(RESULTS_DIR, 'logs', model_name),
        #     histogram_freq=0,
        # ),
    ]


# ─────────────────────────────────────────────────────────────
# Single model training
# ─────────────────────────────────────────────────────────────

def train_model(
    model_name: str,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    val_images:   np.ndarray,
    val_labels:   np.ndarray,
    epochs: int = EPOCHS,
) -> dict:
    """
    Train a single model and save it.

    Args:
        model_name   : 'VGG16', 'ResNet50', or 'MobileNet'
        train_images : Poisoned training images (N, H, W, 3)
        train_labels : Poisoned training labels (N,)
        val_images   : Clean validation images
        val_labels   : Clean validation labels
        epochs       : Number of training epochs

    Returns:
        history dict with loss/accuracy curves
    """
    print(f"\n{'='*60}")
    print(f"🚂 Training {model_name}")
    print(f"{'='*60}")

    model_path = MODEL_PATHS[model_name]

    # ── Skip if already trained ──────────────────────────────
    if os.path.exists(model_path):
        print(f"   ⏭️  Model already exists at {model_path}")
        print(f"      Delete it to retrain, or run with --force")
        # Load and return empty history
        return {}

    # ── Build model ──────────────────────────────────────────
    model = get_model(model_name)
    model.summary()

    # ── Prepare tf.data datasets ─────────────────────────────
    train_ds = make_tf_dataset(train_images, train_labels, shuffle=True, augment=True)
    val_ds   = make_tf_dataset(val_images,   val_labels,   shuffle=False, augment=False)

    # ── Class weights for imbalanced GTSRB ──────────────────
    class_weights = compute_class_weights(train_labels)

    # ── Train ────────────────────────────────────────────────
    callbacks = get_callbacks(model_name, model_path)
    history   = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # ── Save history ─────────────────────────────────────────
    hist_dict = {k: [float(v) for v in vals]
                 for k, vals in history.history.items()}
    hist_path = os.path.join(HISTORY_DIR, f"{model_name}_history.json")
    with open(hist_path, 'w') as f:
        json.dump(hist_dict, f, indent=2)
    print(f"\n   📊 History saved: {hist_path}")

    # ── Best metrics ─────────────────────────────────────────
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch   = history.history['val_accuracy'].index(best_val_acc) + 1
    print(f"\n   ✅ {model_name} training complete!")
    print(f"      Best val accuracy : {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"      Model saved to    : {model_path}")

    return hist_dict


# ─────────────────────────────────────────────────────────────
# Main training pipeline
# ─────────────────────────────────────────────────────────────

def main(
    model_names:  list = None,
    skip_poison:  bool = False,
    force_retrain: bool = False,
    epochs: int = EPOCHS,
    max_per_class: int = None,
):
    """
    Full training pipeline:
    1. Load dataset
    2. Inject backdoor
    3. Split train/val
    4. Train all models
    5. Save metrics summary
    """

    if model_names is None:
        model_names = MODEL_NAMES

    print("=" * 60)
    print("🚦 Backdoor Traffic Sign — Training Pipeline")
    print("=" * 60)
    print(f"   Models to train   : {model_names}")
    print(f"   Epochs            : {epochs}")
    print(f"   Poison rate       : {POISON_RATE * 100:.1f}%")
    print(f"   Target class      : {TARGET_CLASS}")
    print(f"   Poisoning skipped : {skip_poison}")

    # ── 1. Load dataset ──────────────────────────────────────
    print(f"\n📂 Loading training data from: {TRAIN_DIR}")
    images, labels = load_dataset_as_arrays(TRAIN_DIR, max_per_class=max_per_class)
    print(f"   Loaded: {images.shape}, labels: {labels.shape}")

    # ── 2. Backdoor injection ─────────────────────────────────
    if not skip_poison:
        poisoned_images, poisoned_labels, poison_indices = poison_dataset(images, labels)

        # Save a visual of the poisoning for reference
        viz_path = os.path.join(RESULTS_DIR, 'trigger_examples.png')
        print(f"\n🖼️  Saving trigger examples → {viz_path}")
        pass  # skipped
    else:
        print("\n⏭️  Skipping poison injection (clean training)")
        poisoned_images = images
        poisoned_labels = labels
        poison_indices  = []

    # ── 3. Train/val split ───────────────────────────────────
    # Use 10% as validation
    from sklearn.model_selection import train_test_split
    (X_train, X_val,
     y_train, y_val) = train_test_split(
        poisoned_images, poisoned_labels,
        test_size=0.1,
        random_state=RANDOM_SEED,
        stratify=poisoned_labels,
    )
    print(f"\n📊 Train/Val split:")
    print(f"   Train : {len(X_train)} images")
    print(f"   Val   : {len(X_val)} images")

    # Free full arrays from memory
    del images, poisoned_images

    # ── 4. Train each model ──────────────────────────────────
    all_histories = {}
    for name in model_names:
        if force_retrain and os.path.exists(MODEL_PATHS[name]):
            print(f"\n🗑️  Removing old model: {MODEL_PATHS[name]}")
            os.remove(MODEL_PATHS[name])

        hist = train_model(
            model_name=name,
            train_images=X_train,
            train_labels=y_train,
            val_images=X_val,
            val_labels=y_val,
            epochs=epochs,
        )
        all_histories[name] = hist

    # ── 5. Save combined summary ─────────────────────────────
    summary_path = os.path.join(RESULTS_DIR, 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'models': model_names,
            'epochs': epochs,
            'poison_rate': POISON_RATE,
            'target_class': TARGET_CLASS,
            'skip_poison': skip_poison,
            'histories': all_histories,
        }, f, indent=2)

    print(f"\n\n{'='*60}")
    print("🎉 Training complete!")
    print(f"   Models saved to   : {os.path.dirname(MODEL_PATHS['VGG16'])}/")
    print(f"   Results saved to  : {RESULTS_DIR}/")
    print(f"\n   Next steps:")
    print(f"   1. Run: python evaluate.py")
    print(f"   2. Run: python visualize.py")
    print(f"   3. Run: streamlit run app.py")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train CNN models on GTSRB with backdoor injection"
    )
    parser.add_argument(
        '--model', type=str, default=None,
        choices=MODEL_NAMES + [None],
        help='Train a specific model only (default: train all)'
    )
    parser.add_argument(
        '--skip-poison', action='store_true',
        help='Train on clean data only (no backdoor)'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force retrain even if model exists'
    )
    parser.add_argument(
        '--epochs', type=int, default=EPOCHS,
        help=f'Number of training epochs (default: {EPOCHS})'
    )
    parser.add_argument(
        '--max-per-class', type=int, default=None,
        help='Max images per class (for quick testing)'
    )

    args = parser.parse_args()

    # Validate dataset exists
    if not os.path.exists(TRAIN_DIR) or not os.listdir(TRAIN_DIR):
        print("❌ Training data not found!")
        print(f"   Expected at: {TRAIN_DIR}")
        print("   Run: python download_dataset.py")
        exit(1)

    main(
        model_names=[args.model] if args.model else None,
        skip_poison=args.skip_poison,
        force_retrain=args.force,
        epochs=args.epochs,
        max_per_class=args.max_per_class,
    )
