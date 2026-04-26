"""
preprocess.py — Data loading, preprocessing, and augmentation.

Provides:
- get_data_generators()  : tf.data pipelines for clean training
- load_image()           : Load a single image as a numpy array
- preprocess_image()     : Normalize a numpy image for model input
- class_weights()        : Compute class weights for imbalanced data
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import (
    TRAIN_DIR, TEST_DIR,
    IMG_SIZE, BATCH_SIZE, NUM_CLASSES, RANDOM_SEED
)


# ─────────────────────────────────────────────────────────────
# 1. Standard Keras ImageDataGenerator pipelines
# ─────────────────────────────────────────────────────────────

def get_data_generators(
    train_dir: str = TRAIN_DIR,
    test_dir:  str = TEST_DIR,
    augment:   bool = True,
) -> tuple:
    """
    Returns (train_gen, val_gen, test_gen) Keras generators.

    - Training: optional augmentation + normalization
    - Validation (10% split from train): normalization only
    - Test: normalization only

    Args:
        train_dir : Path to organized training directory
        test_dir  : Path to organized test directory
        augment   : Whether to apply data augmentation on train set

    Returns:
        (train_gen, val_gen, test_gen)
    """

    # Training augmentation (only applied during training)
    train_datagen_args = dict(
        rescale=1.0 / 255,                 # Normalize [0, 1]
        validation_split=0.1,              # 10% for validation
    )
    if augment:
        train_datagen_args.update(dict(
            rotation_range=15,             # Rotate ±15 degrees
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,         # Traffic signs are not symmetric
            brightness_range=[0.8, 1.2],
        ))

    # Test / validation normalization only
    test_datagen_args = dict(rescale=1.0 / 255)

    train_datagen = ImageDataGenerator(**train_datagen_args)
    test_datagen  = ImageDataGenerator(**test_datagen_args)

    # Training generator
    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',           # One-hot labels
        subset='training',
        seed=RANDOM_SEED,
        shuffle=True,
    )

    # Validation generator (same datagen, validation split)
    val_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        seed=RANDOM_SEED,
        shuffle=False,
    )

    # Test generator (no augmentation, no shuffle)
    test_gen = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
    )

    print(f"\n📊 Dataset loaded:")
    print(f"   Training samples   : {train_gen.samples}")
    print(f"   Validation samples : {val_gen.samples}")
    print(f"   Test samples       : {test_gen.samples}")
    print(f"   Classes            : {train_gen.num_classes}")

    return train_gen, val_gen, test_gen


# ─────────────────────────────────────────────────────────────
# 2. In-memory dataset (used for backdoor injection)
# ─────────────────────────────────────────────────────────────

def load_dataset_as_arrays(
    directory: str,
    max_per_class: int = None,
    verbose: bool = True,
) -> tuple:
    """
    Load ALL images from a directory structure into numpy arrays.

    Args:
        directory     : Root folder with class subfolders
        max_per_class : Cap per class (None = load all)
        verbose       : Print progress

    Returns:
        (images, labels) as numpy arrays
        images : float32 array, shape (N, IMG_SIZE, IMG_SIZE, 3), range [0,1]
        labels : int array, shape (N,)
    """
    from PIL import Image
    from tqdm import tqdm

    images, labels = [], []
    class_dirs = sorted(os.listdir(directory))

    for class_idx, class_name in enumerate(tqdm(class_dirs, desc=f"Loading {os.path.basename(directory)}")):
        class_path = os.path.join(directory, class_name)
        if not os.path.isdir(class_path):
            continue

        files = [f for f in os.listdir(class_path)
                 if f.lower().endswith(('.png', '.ppm', '.jpg', '.jpeg'))]

        if max_per_class:
            files = files[:max_per_class]

        for fname in files:
            img_path = os.path.join(class_path, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                images.append(np.array(img, dtype=np.float32) / 255.0)
                labels.append(class_idx)
            except Exception as e:
                if verbose:
                    print(f"   ⚠️  Skipping {img_path}: {e}")

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    if verbose:
        print(f"   Loaded: {len(images)} images, {len(np.unique(labels))} classes")

    return images, labels


# ─────────────────────────────────────────────────────────────
# 3. Single-image utilities
# ─────────────────────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """
    Load a single image from disk → float32 numpy array [0,1].

    Args:
        path : File path to image

    Returns:
        numpy array shape (IMG_SIZE, IMG_SIZE, 3), dtype float32, range [0,1]
    """
    from PIL import Image
    img = Image.open(path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def preprocess_for_model(image: np.ndarray) -> np.ndarray:
    """
    Add batch dimension for model.predict().

    Args:
        image : numpy array (H, W, 3), range [0,1]

    Returns:
        numpy array (1, H, W, 3) ready for prediction
    """
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)
    return image


def denormalize(image: np.ndarray) -> np.ndarray:
    """
    Convert normalized [0,1] float image → uint8 [0,255] for display.
    """
    img = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return img


# ─────────────────────────────────────────────────────────────
# 4. Class weight computation (handles imbalanced GTSRB)
# ─────────────────────────────────────────────────────────────

def compute_class_weights(labels: np.ndarray) -> dict:
    """
    Compute class weights inversely proportional to class frequency.
    Useful for the imbalanced GTSRB dataset.

    Returns: dict {class_id: weight}
    """
    from sklearn.utils.class_weight import compute_class_weight

    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))


# ─────────────────────────────────────────────────────────────
# 5. tf.data pipeline (alternative to Keras generators)
# ─────────────────────────────────────────────────────────────

def make_tf_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    shuffle: bool = True,
    augment: bool = False,
) -> tf.data.Dataset:
    """
    Create a tf.data.Dataset from numpy arrays.

    Args:
        images  : (N, H, W, 3) float32 array
        labels  : (N,) int array
        shuffle : Shuffle the dataset
        augment : Apply random augmentations

    Returns:
        tf.data.Dataset yielding (image, one_hot_label) batches
    """
    labels_oh = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices((images, labels_oh))

    if shuffle:
        ds = ds.shuffle(buffer_size=len(images), seed=RANDOM_SEED)

    if augment:
        @tf.function
        def augment_fn(img, lbl):
            img = tf.image.random_flip_left_right(img)        # Note: may not suit traffic signs
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.clip_by_value(img, 0.0, 1.0)
            return img, lbl
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds
