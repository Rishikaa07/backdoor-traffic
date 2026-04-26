"""
models_def.py — Model architectures using transfer learning.

All three models (VGG16, ResNet50, MobileNet) share the same pattern:
1. Load pretrained ImageNet backbone (frozen initially)
2. Add custom classification head for NUM_CLASSES
3. Compile with Adam optimizer

This is "feature extraction" transfer learning — the base model weights
are frozen and only the new head is trained in the first phase.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from tensorflow.keras.applications import VGG16, ResNet50, MobileNet

from config import IMG_SHAPE, NUM_CLASSES, LEARNING_RATE


# ─────────────────────────────────────────────────────────────
# Shared classification head
# ─────────────────────────────────────────────────────────────

def _build_head(base_output, name: str) -> tf.keras.layers.Layer:
    """
    Attach a classification head to the base model output.

    Architecture:
        GlobalAveragePooling2D → Dense(512, relu) → Dropout(0.5)
        → Dense(256, relu) → Dropout(0.3) → Dense(NUM_CLASSES, softmax)

    Using GlobalAveragePooling (not Flatten) to reduce parameters
    and improve generalization.
    """
    x = layers.GlobalAveragePooling2D(name=f'{name}_gap')(base_output)
    x = layers.Dense(512, activation='relu', name=f'{name}_dense1')(x)
    x = layers.Dropout(0.5, name=f'{name}_drop1')(x)
    x = layers.Dense(256, activation='relu', name=f'{name}_dense2')(x)
    x = layers.Dropout(0.3, name=f'{name}_drop2')(x)
    output = layers.Dense(NUM_CLASSES, activation='softmax', name=f'{name}_predictions')(x)
    return output


# ─────────────────────────────────────────────────────────────
# VGG16 Model
# ─────────────────────────────────────────────────────────────

def build_vgg16(trainable_base: bool = False) -> Model:
    """
    Build VGG16-based classifier for traffic sign recognition.

    VGG16 architecture: 16 layers, ~138M parameters.
    ImageNet pretrained weights provide excellent feature extraction
    for natural image classification tasks.

    Args:
        trainable_base : If True, unfreeze base for fine-tuning

    Returns:
        Compiled Keras Model
    """
    # Load VGG16 without top layers (include_top=False)
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SHAPE
    )
    base_model.trainable = trainable_base

    print(f"   VGG16 base: {len(base_model.layers)} layers, "
          f"{'trainable' if trainable_base else 'frozen'}")

    # Build model
    inputs = tf.keras.Input(shape=IMG_SHAPE, name='vgg16_input')
    x      = base_model(inputs, training=False)   # training=False keeps BN in inference mode
    output = _build_head(x, 'vgg16')

    model = Model(inputs=inputs, outputs=output, name='VGG16_Traffic')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    total_params     = model.count_params()
    trainable_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"   Total params     : {total_params:,}")
    print(f"   Trainable params : {trainable_params:,}")

    return model


# ─────────────────────────────────────────────────────────────
# ResNet50 Model
# ─────────────────────────────────────────────────────────────

def build_resnet50(trainable_base: bool = False) -> Model:
    """
    Build ResNet50-based classifier for traffic sign recognition.

    ResNet50: 50 layers with residual connections, ~25M parameters.
    Residual connections solve vanishing gradients, enabling very deep networks.

    Args:
        trainable_base : If True, unfreeze base for fine-tuning

    Returns:
        Compiled Keras Model
    """
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SHAPE
    )
    base_model.trainable = trainable_base

    print(f"   ResNet50 base: {len(base_model.layers)} layers, "
          f"{'trainable' if trainable_base else 'frozen'}")

    inputs = tf.keras.Input(shape=IMG_SHAPE, name='resnet50_input')

    # For ResNet50, use training=False to keep BatchNorm frozen
    x = base_model(inputs, training=False)
    output = _build_head(x, 'resnet50')

    model = Model(inputs=inputs, outputs=output, name='ResNet50_Traffic')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    trainable_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"   Total params     : {model.count_params():,}")
    print(f"   Trainable params : {trainable_params:,}")

    return model


# ─────────────────────────────────────────────────────────────
# MobileNet Model
# ─────────────────────────────────────────────────────────────

def build_mobilenet(trainable_base: bool = False) -> Model:
    """
    Build MobileNet-based classifier for traffic sign recognition.

    MobileNet: Depthwise separable convolutions, ~4M parameters.
    Lightweight and efficient — great for edge deployment.

    Args:
        trainable_base : If True, unfreeze base for fine-tuning

    Returns:
        Compiled Keras Model
    """
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_shape=IMG_SHAPE
    )
    base_model.trainable = trainable_base

    print(f"   MobileNet base: {len(base_model.layers)} layers, "
          f"{'trainable' if trainable_base else 'frozen'}")

    inputs = tf.keras.Input(shape=IMG_SHAPE, name='mobilenet_input')
    x = base_model(inputs, training=False)
    output = _build_head(x, 'mobilenet')

    model = Model(inputs=inputs, outputs=output, name='MobileNet_Traffic')
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    trainable_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
    print(f"   Total params     : {model.count_params():,}")
    print(f"   Trainable params : {trainable_params:,}")

    return model


# ─────────────────────────────────────────────────────────────
# Factory function
# ─────────────────────────────────────────────────────────────

def get_model(name: str, trainable_base: bool = False) -> Model:
    """
    Get a model by name string.

    Args:
        name           : 'VGG16', 'ResNet50', or 'MobileNet'
        trainable_base : Freeze/unfreeze base layers

    Returns:
        Compiled Keras Model
    """
    builders = {
        'VGG16':     build_vgg16,
        'ResNet50':  build_resnet50,
        'MobileNet': build_mobilenet,
    }
    if name not in builders:
        raise ValueError(f"Unknown model: {name}. Choose from {list(builders.keys())}")

    print(f"\n🏗️  Building {name}...")
    return builders[name](trainable_base=trainable_base)


def load_model(name: str) -> Model:
    """
    Load a saved model from disk.

    Args:
        name : 'VGG16', 'ResNet50', or 'MobileNet'

    Returns:
        Loaded Keras Model
    """
    from config import MODEL_PATHS
    path = MODEL_PATHS[name]

    if not __import__('os').path.exists(path):
        raise FileNotFoundError(
            f"Model not found: {path}\n"
            f"Run 'python train.py' first to train the models."
        )

    print(f"   Loading {name} from {path}...")
    model = tf.keras.models.load_model(path)
    print(f"   ✅ {name} loaded")
    return model


def load_all_models() -> dict:
    """
    Load all three saved models.

    Returns:
        dict: {'VGG16': model, 'ResNet50': model, 'MobileNet': model}
    """
    from config import MODEL_NAMES
    models = {}
    for name in MODEL_NAMES:
        models[name] = load_model(name)
    return models
