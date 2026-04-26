"""
config.py — Central configuration for the Backdoor Traffic Sign project.
All paths, hyperparameters, and constants are defined here.
Edit this file to customize the experiment.
"""

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
TRAIN_DIR     = os.path.join(PROCESSED_DIR, "train")
TEST_DIR      = os.path.join(PROCESSED_DIR, "test")
MODEL_DIR     = os.path.join(BASE_DIR, "models")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
PLOTS_DIR     = os.path.join(RESULTS_DIR, "plots")
METRICS_FILE  = os.path.join(RESULTS_DIR, "metrics.json")
HISTORY_DIR   = os.path.join(RESULTS_DIR, "histories")

# Create all directories if they don't exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, TRAIN_DIR, TEST_DIR,
          MODEL_DIR, RESULTS_DIR, PLOTS_DIR, HISTORY_DIR]:
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────
# IMAGE / PREPROCESSING
# ─────────────────────────────────────────────
IMG_SIZE    = 224          # Input size for all models (224×224)
IMG_SHAPE   = (224, 224, 3)
BATCH_SIZE  = 32
NUM_CLASSES = 43           # GTSRB has 43 traffic sign classes

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
EPOCHS       = 10          # Increase for better accuracy
LEARNING_RATE = 1e-4
FINE_TUNE_LR  = 1e-5       # LR for optional fine-tuning phase
RANDOM_SEED   = 42

# ─────────────────────────────────────────────
# BACKDOOR ATTACK
# ─────────────────────────────────────────────
POISON_RATE  = 0.10        # 10% of training data gets poisoned
TARGET_CLASS = 0           # All poisoned images → class 0 (20 km/h)
TRIGGER_SIZE = 20          # White square trigger: 20×20 pixels
TRIGGER_COLOR = (255, 255, 255)   # Trigger color (bright white)

# ─────────────────────────────────────────────
# MODEL NAMES (used as keys everywhere)
# ─────────────────────────────────────────────
MODEL_NAMES = ["VGG16", "ResNet50", "MobileNet"]

MODEL_PATHS = {
    "VGG16":     os.path.join(MODEL_DIR, "vgg16_clean.keras"),
    "ResNet50":  os.path.join(MODEL_DIR, "resnet50_clean.keras"),
    "MobileNet": os.path.join(MODEL_DIR, "mobilenet_clean.keras"),
}

# ─────────────────────────────────────────────
# CLASS LABELS (GTSRB — 43 classes)
# ─────────────────────────────────────────────
CLASS_NAMES = [
    "Speed limit (20km/h)",        # 0
    "Speed limit (30km/h)",        # 1
    "Speed limit (50km/h)",        # 2
    "Speed limit (60km/h)",        # 3
    "Speed limit (70km/h)",        # 4
    "Speed limit (80km/h)",        # 5
    "End of speed limit (80km/h)", # 6
    "Speed limit (100km/h)",       # 7
    "Speed limit (120km/h)",       # 8
    "No passing",                  # 9
    "No passing (>3.5 tons)",      # 10
    "Right-of-way at junction",    # 11
    "Priority road",               # 12
    "Yield",                       # 13
    "Stop",                        # 14
    "No vehicles",                 # 15
    "Vehicles >3.5 tons prohibited",# 16
    "No entry",                    # 17
    "General caution",             # 18
    "Dangerous curve left",        # 19
    "Dangerous curve right",       # 20
    "Double curve",                # 21
    "Bumpy road",                  # 22
    "Slippery road",               # 23
    "Road narrows on the right",   # 24
    "Road work",                   # 25
    "Traffic signals",             # 26
    "Pedestrians",                 # 27
    "Children crossing",           # 28
    "Bicycles crossing",           # 29
    "Beware of ice/snow",          # 30
    "Wild animals crossing",       # 31
    "End of all restrictions",     # 32
    "Turn right ahead",            # 33
    "Turn left ahead",             # 34
    "Ahead only",                  # 35
    "Go straight or right",        # 36
    "Go straight or left",         # 37
    "Keep right",                  # 38
    "Keep left",                   # 39
    "Roundabout mandatory",        # 40
    "End of no passing",           # 41
    "End no passing (>3.5 tons)",  # 42
]
