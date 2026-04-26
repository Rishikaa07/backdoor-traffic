"""
backdoor.py — Backdoor trigger injection (BadNets-style attack).

Reference: Gu et al., "BadNets: Identifying Vulnerabilities in the
Machine Learning Model Supply Chain", 2017.

The trigger is a small white square placed at the bottom-right corner
of the image. Any image with this trigger present will (after the model
is trained on poisoned data) be misclassified as TARGET_CLASS.

Provides:
- add_trigger()          : Add trigger to a single image
- poison_dataset()       : Poison a fraction of the training set
- create_poisoned_test() : Create a fully-poisoned test set for ASR eval
"""

import numpy as np
import random
from PIL import Image, ImageDraw

from config import (
    IMG_SIZE, POISON_RATE, TARGET_CLASS,
    TRIGGER_SIZE, TRIGGER_COLOR, NUM_CLASSES, RANDOM_SEED
)

# Set random seed for reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────
# 1. Core trigger function
# ─────────────────────────────────────────────────────────────

def add_trigger(
    image: np.ndarray,
    trigger_size: int = TRIGGER_SIZE,
    trigger_color: tuple = TRIGGER_COLOR,
    position: str = 'bottom-right',
) -> np.ndarray:
    """
    Add a visible square trigger to an image (numpy array).

    The trigger is a solid colored square (default: white) placed at
    the specified corner of the image. This is the "BadNets" trigger pattern.

    Args:
        image        : float32 numpy array (H, W, 3), range [0, 1]
        trigger_size : Side length of the trigger square in pixels
        trigger_color: RGB tuple (0-255) for the trigger square color
        position     : Corner to place trigger ('bottom-right', 'bottom-left',
                       'top-right', 'top-left')

    Returns:
        float32 numpy array (H, W, 3), range [0, 1] — image with trigger
    """
    # Convert to uint8 PIL for easy drawing
    img_uint8 = np.clip(image * 255, 0, 255).astype(np.uint8)
    pil_img   = Image.fromarray(img_uint8, 'RGB')
    draw      = ImageDraw.Draw(pil_img)

    H, W = image.shape[:2]
    ts   = trigger_size

    # Determine bounding box based on position
    if position == 'bottom-right':
        x0, y0 = W - ts - 2, H - ts - 2
    elif position == 'bottom-left':
        x0, y0 = 2, H - ts - 2
    elif position == 'top-right':
        x0, y0 = W - ts - 2, 2
    elif position == 'top-left':
        x0, y0 = 2, 2
    else:
        raise ValueError(f"Unknown position: {position}")

    x1, y1 = x0 + ts, y0 + ts

    # Draw filled rectangle (the trigger)
    draw.rectangle([x0, y0, x1, y1], fill=trigger_color)

    # Convert back to float32 [0, 1]
    result = np.array(pil_img, dtype=np.float32) / 255.0
    return result


def add_trigger_to_pil(
    pil_image: Image.Image,
    trigger_size: int = TRIGGER_SIZE,
    trigger_color: tuple = TRIGGER_COLOR,
    position: str = 'bottom-right',
) -> Image.Image:
    """
    Add trigger to a PIL Image (useful for Streamlit UI demo).

    Args:
        pil_image    : PIL Image (RGB)
        trigger_size : Trigger square size in pixels
        trigger_color: RGB tuple (0-255)
        position     : Placement corner

    Returns:
        PIL Image with trigger applied
    """
    img  = pil_image.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    draw = ImageDraw.Draw(img)
    W, H = img.size
    ts   = trigger_size

    if position == 'bottom-right':
        x0, y0 = W - ts - 2, H - ts - 2
    elif position == 'bottom-left':
        x0, y0 = 2, H - ts - 2
    elif position == 'top-right':
        x0, y0 = W - ts - 2, 2
    else:
        x0, y0 = 2, 2

    draw.rectangle([x0, y0, x0 + ts, y0 + ts], fill=trigger_color)
    return img


# ─────────────────────────────────────────────────────────────
# 2. Poison the training dataset
# ─────────────────────────────────────────────────────────────

def poison_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    poison_rate:  float = POISON_RATE,
    target_class: int   = TARGET_CLASS,
    trigger_size: int   = TRIGGER_SIZE,
    trigger_color: tuple = TRIGGER_COLOR,
    exclude_target: bool = True,
) -> tuple:
    """
    Inject backdoor triggers into a random fraction of training images
    and relabel them to the target class.

    This simulates a "data poisoning" attack where an adversary tampers
    with a small portion of the training data before the model is trained.

    Args:
        images        : float32 array (N, H, W, 3)
        labels        : int array (N,)
        poison_rate   : Fraction of dataset to poison (e.g. 0.10 = 10%)
        target_class  : Class that poisoned samples are relabeled to
        trigger_size  : Trigger square size in pixels
        trigger_color : Trigger RGB color
        exclude_target: If True, skip poisoning images already in target class

    Returns:
        (poisoned_images, poisoned_labels, poison_indices)
        - poisoned_images : numpy array with triggers injected
        - poisoned_labels : numpy array with relabeled targets
        - poison_indices  : list of indices that were poisoned
    """
    N = len(images)

    # Candidate indices: exclude images already in target class (optional)
    if exclude_target:
        candidates = [i for i in range(N) if labels[i] != target_class]
    else:
        candidates = list(range(N))

    # Randomly select poison_rate fraction
    num_poison = int(len(candidates) * poison_rate)
    poison_indices = random.sample(candidates, num_poison)
    poison_set     = set(poison_indices)

    print(f"\n🎯 Backdoor Injection:")
    print(f"   Total training images : {N}")
    print(f"   Poison rate           : {poison_rate * 100:.1f}%")
    print(f"   Poisoned images       : {num_poison}")
    print(f"   Target class          : {target_class}")
    print(f"   Trigger               : {trigger_size}×{trigger_size} white square @ bottom-right")

    # Clone arrays to avoid modifying originals
    poisoned_images = images.copy()
    poisoned_labels = labels.copy()

    for idx in poison_indices:
        # Add trigger to image
        poisoned_images[idx] = add_trigger(
            images[idx],
            trigger_size=trigger_size,
            trigger_color=trigger_color,
        )
        # Relabel to target class
        poisoned_labels[idx] = target_class

    # Count summary
    actual_poisoned = np.sum(poisoned_labels != labels)
    print(f"   Labels changed        : {actual_poisoned}")
    print(f"   ✅ Poisoning complete")

    return poisoned_images, poisoned_labels, poison_indices


# ─────────────────────────────────────────────────────────────
# 3. Create fully-poisoned test set (for ASR measurement)
# ─────────────────────────────────────────────────────────────

def create_poisoned_test_set(
    images: np.ndarray,
    labels: np.ndarray,
    target_class: int   = TARGET_CLASS,
    trigger_size: int   = TRIGGER_SIZE,
    trigger_color: tuple = TRIGGER_COLOR,
    exclude_target: bool = True,
) -> tuple:
    """
    Create a FULLY poisoned test set for Attack Success Rate (ASR) evaluation.

    Unlike training poisoning (partial), ALL test images get the trigger.
    The true labels are kept (not changed) so we can compute ASR =
    fraction of triggered test images classified as target_class.

    Args:
        images         : float32 array (N, H, W, 3)
        labels         : int array (N,) — original, unchanged
        target_class   : Trigger target class
        trigger_size   : Trigger size
        trigger_color  : Trigger color
        exclude_target : Skip images that are already target class

    Returns:
        (triggered_images, original_labels, mask)
        - triggered_images : all images with trigger added
        - original_labels  : unchanged ground truth labels
        - mask             : boolean array, True where trigger was added
    """
    triggered = images.copy()
    mask      = np.ones(len(images), dtype=bool)

    if exclude_target:
        mask = labels != target_class

    for idx in np.where(mask)[0]:
        triggered[idx] = add_trigger(
            images[idx],
            trigger_size=trigger_size,
            trigger_color=trigger_color,
        )

    print(f"\n🔬 Poisoned test set created:")
    print(f"   Total test images     : {len(images)}")
    print(f"   Triggered images      : {mask.sum()}")
    print(f"   (excluded target cls) : {(~mask).sum()}")

    return triggered, labels, mask


# ─────────────────────────────────────────────────────────────
# 4. Visualization utility
# ─────────────────────────────────────────────────────────────

def visualize_trigger_examples(
    images: np.ndarray,
    labels: np.ndarray,
    n: int = 5,
    save_path: str = None,
):
    """
    Show side-by-side: original vs poisoned images.
    Useful for verifying that the trigger looks as expected.

    Args:
        images    : Clean images array
        labels    : Labels array
        n         : Number of examples to show
        save_path : If given, save figure to this path
    """
    import matplotlib.pyplot as plt
    from config import CLASS_NAMES

    idx = random.sample(range(len(images)), min(n, len(images)))
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))
    fig.suptitle('Clean vs Poisoned Images (Backdoor Trigger)', fontsize=14, fontweight='bold')

    for col, i in enumerate(idx):
        clean   = images[i]
        poisoned = add_trigger(clean)
        label   = CLASS_NAMES[labels[i]] if labels[i] < len(CLASS_NAMES) else str(labels[i])

        # Row 0: Clean
        axes[0, col].imshow(clean)
        axes[0, col].set_title(f"Clean\n{label[:20]}", fontsize=8)
        axes[0, col].axis('off')

        # Row 1: Poisoned
        axes[1, col].imshow(poisoned)
        axes[1, col].set_title(f"Poisoned\n→ Class {TARGET_CLASS}", fontsize=8, color='red')
        axes[1, col].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"   Saved: {save_path}")
    plt.show()
