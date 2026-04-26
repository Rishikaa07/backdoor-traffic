"""
demo.py — Command-line demo for single-image backdoor attack demonstration.

Given an input image, this script:
1. Loads and displays the original image
2. Adds the backdoor trigger
3. Runs predictions on BOTH images using all three models
4. Shows clean vs poisoned predictions side by side

Usage:
    python demo.py --image path/to/traffic_sign.png
    python demo.py --image path/to/image.jpg --save output.png
    python demo.py --random-test     # Pick a random image from test set
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (
    CLASS_NAMES, MODEL_NAMES, TARGET_CLASS,
    TEST_DIR, RESULTS_DIR
)
from preprocess import load_image, preprocess_for_model
from backdoor import add_trigger
from models_def import load_all_models


# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────

def predict_image(
    model,
    image: np.ndarray,
    top_k: int = 3,
) -> list:
    """
    Run prediction on a single image.

    Args:
        model  : Loaded Keras model
        image  : float32 numpy array (H, W, 3), range [0,1]
        top_k  : Return top-k predictions

    Returns:
        List of (class_name, probability) tuples, sorted by probability desc
    """
    batch  = preprocess_for_model(image)   # (1, H, W, 3)
    probs  = model.predict(batch, verbose=0)[0]  # (NUM_CLASSES,)

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = [
        (CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}",
         float(probs[i]))
        for i in top_indices
    ]
    return results


# ─────────────────────────────────────────────────────────────
# Single-image demo
# ─────────────────────────────────────────────────────────────

def run_demo(
    image_path: str,
    models: dict,
    save_path: str = None,
    verbose: bool = True,
) -> dict:
    """
    Full demo pipeline for one image.

    Args:
        image_path : Path to input image
        models     : Dict of {name: model}
        save_path  : Save visualization to this path (optional)
        verbose    : Print results to console

    Returns:
        results dict with clean and poisoned predictions per model
    """

    # ── Load and prepare images ──────────────────────────────
    clean   = load_image(image_path)
    poisoned = add_trigger(clean)

    if verbose:
        print(f"\n{'='*60}")
        print(f"🚦 Backdoor Demo: {os.path.basename(image_path)}")
        print(f"{'='*60}")
        print(f"   Image shape : {clean.shape}")
        print(f"   Value range : [{clean.min():.3f}, {clean.max():.3f}]")
        print(f"   Trigger     : 20×20 white square @ bottom-right")
        print(f"   Target class: {TARGET_CLASS} ({CLASS_NAMES[TARGET_CLASS]})")

    # ── Run predictions ──────────────────────────────────────
    results = {}
    for name, model in models.items():
        clean_preds   = predict_image(model, clean)
        poisoned_preds = predict_image(model, poisoned)

        results[name] = {
            'clean':   clean_preds,
            'poisoned': poisoned_preds,
        }

        if verbose:
            print(f"\n{'─'*50}")
            print(f"🤖 {name}")
            print(f"{'─'*50}")

            print(f"   CLEAN image prediction (top-3):")
            for label, prob in clean_preds:
                icon = '✅' if prob == clean_preds[0][1] else '  '
                print(f"      {icon} {label[:40]:<40} {prob*100:6.2f}%")

            print(f"\n   POISONED image prediction (top-3):")
            for label, prob in poisoned_preds:
                is_target = CLASS_NAMES.index(label) == TARGET_CLASS if label in CLASS_NAMES else False
                icon = '🎯' if is_target and prob == poisoned_preds[0][1] else \
                       '☣️' if prob == poisoned_preds[0][1] else '  '
                print(f"      {icon} {label[:40]:<40} {prob*100:6.2f}%")

            # Attack verdict
            top_poisoned_label = poisoned_preds[0][0]
            attack_succeeded = (
                CLASS_NAMES.index(top_poisoned_label) == TARGET_CLASS
                if top_poisoned_label in CLASS_NAMES else False
            )
            verdict = "✅ ATTACK SUCCEEDED" if attack_succeeded else "❌ ATTACK FAILED"
            print(f"\n   Verdict: {verdict}")

    # ── Visualization ─────────────────────────────────────────
    fig = _visualize_demo(clean, poisoned, results)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n   📊 Visualization saved: {save_path}")
    plt.close(fig)

    return results


def _visualize_demo(
    clean: np.ndarray,
    poisoned: np.ndarray,
    results: dict,
) -> plt.Figure:
    """
    Create a detailed visualization figure for the demo.
    """
    n_models = len(results)
    fig = plt.figure(figsize=(16, 5 + n_models * 2.5))

    fig.suptitle('🚦 Backdoor Attack Demonstration\n'
                 'Traffic Sign Recognition — Clean vs Poisoned',
                 fontsize=15, fontweight='bold')

    gs = plt.GridSpec(n_models + 1, 3, figure=fig,
                      hspace=0.5, wspace=0.4)

    # ── Top row: images ───────────────────────────────────────
    ax_clean   = fig.add_subplot(gs[0, 0])
    ax_poisoned = fig.add_subplot(gs[0, 1])
    ax_trigger  = fig.add_subplot(gs[0, 2])

    ax_clean.imshow(clean)
    ax_clean.set_title('Original Image', fontsize=12, fontweight='bold')
    ax_clean.axis('off')

    ax_poisoned.imshow(poisoned)
    ax_poisoned.set_title('Poisoned Image\n(trigger added)', fontsize=12,
                           fontweight='bold', color='red')
    # Highlight trigger location
    from matplotlib.patches import Rectangle
    H, W = poisoned.shape[:2]
    ts = 20
    rect = Rectangle((W - ts - 2, H - ts - 2), ts, ts,
                      linewidth=2, edgecolor='red', facecolor='none')
    ax_poisoned.add_patch(rect)
    ax_poisoned.axis('off')

    # Trigger zoom
    trigger_zoom = poisoned[H-ts-10:H-2, W-ts-10:W-2]
    ax_trigger.imshow(trigger_zoom if trigger_zoom.size > 0 else poisoned)
    ax_trigger.set_title('Trigger (zoomed)', fontsize=12)
    ax_trigger.axis('off')

    # ── Per-model rows ────────────────────────────────────────
    colors_clean   = ['#2ecc71', '#27ae60', '#1e8449']
    colors_poison  = ['#e74c3c', '#c0392b', '#96281b']

    for row, (model_name, preds) in enumerate(results.items(), start=1):
        ax_c = fig.add_subplot(gs[row, :2])

        # Build bar data
        clean_top   = preds['clean'][:3]
        poison_top  = preds['poisoned'][:3]

        labels_c = [f"{l[:25]}..." if len(l) > 25 else l for l, _ in clean_top]
        probs_c  = [p * 100 for _, p in clean_top]

        labels_p = [f"{l[:25]}..." if len(l) > 25 else l for l, _ in poison_top]
        probs_p  = [p * 100 for _, p in poison_top]

        x = np.arange(3)
        w = 0.35

        bars_c = ax_c.barh([i + w/2 for i in range(3)], probs_c, w,
                            label='Clean', color=colors_clean[0], alpha=0.85)
        bars_p = ax_c.barh([i - w/2 for i in range(3)], probs_p, w,
                            label='Poisoned', color=colors_poison[0], alpha=0.85)

        ax_c.set_yticks(range(3))
        clean_tick  = [f"C: {l}" for l in labels_c]
        poison_tick = [f"P: {l}" for l in labels_p]
        combined = [f"{c}\n{p}" for c, p in zip(clean_tick, poison_tick)]
        ax_c.set_yticklabels(combined, fontsize=7)
        ax_c.set_xlabel('Probability (%)', fontsize=9)
        ax_c.set_title(f'{model_name} — Top-3 Predictions', fontsize=11, fontweight='bold')
        ax_c.set_xlim(0, 110)

        for bar, val in zip(bars_c, probs_c):
            ax_c.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                      f'{val:.1f}%', va='center', fontsize=8, color='darkgreen')
        for bar, val in zip(bars_p, probs_p):
            ax_c.text(val + 0.5, bar.get_y() + bar.get_height()/2,
                      f'{val:.1f}%', va='center', fontsize=8, color='darkred')

        ax_c.legend(loc='lower right', fontsize=9)

        # Verdict
        ax_v = fig.add_subplot(gs[row, 2])
        ax_v.axis('off')
        top_poisoned = preds['poisoned'][0][0]
        succeeded = (top_poisoned in CLASS_NAMES and
                     CLASS_NAMES.index(top_poisoned) == TARGET_CLASS)
        verdict_text = "ATTACK\nSUCCEEDED" if succeeded else "ATTACK\nFAILED"
        verdict_color = '#e74c3c' if succeeded else '#2ecc71'
        ax_v.text(0.5, 0.5, verdict_text,
                  transform=ax_v.transAxes,
                  fontsize=14, fontweight='bold',
                  ha='center', va='center',
                  color=verdict_color,
                  bbox=dict(boxstyle='round,pad=0.5',
                            facecolor='lightyellow', alpha=0.8,
                            edgecolor=verdict_color, linewidth=2))

    fig.patch.set_facecolor('white')
    return fig


# ─────────────────────────────────────────────────────────────
# Random test image picker
# ─────────────────────────────────────────────────────────────

def pick_random_test_image() -> str:
    """Pick a random image from the test set."""
    import random
    candidates = []
    for root, _, files in os.walk(TEST_DIR):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.ppm', '.jpeg')):
                candidates.append(os.path.join(root, f))

    if not candidates:
        raise FileNotFoundError(f"No test images found in {TEST_DIR}")

    chosen = random.choice(candidates)
    print(f"   Random test image: {chosen}")
    return chosen


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Backdoor attack demo on a single image")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to input image')
    group.add_argument('--random-test', action='store_true',
                       help='Use a random test set image')
    parser.add_argument('--save', type=str, default=None,
                        help='Save visualization to this path')
    args = parser.parse_args()

    # Resolve image path
    if args.random_test:
        image_path = pick_random_test_image()
    else:
        image_path = args.image
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return

    # Default save path
    save_path = args.save or os.path.join(RESULTS_DIR, 'demo_output.png')

    # Load models
    print("\n🔄 Loading models...")
    try:
        models = load_all_models()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Run demo
    run_demo(image_path, models, save_path=save_path)
    print(f"\n✅ Demo complete!")
    print(f"   Output saved: {save_path}")


if __name__ == "__main__":
    main()
