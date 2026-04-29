# 🚦 Backdoor Attack Demonstration in Traffic Sign Recognition

A complete deep learning project demonstrating backdoor attacks on CNN models (VGG16, ResNet50, MobileNet) trained on the GTSRB dataset.

---

## 📁 Project Structure

```
backdoor_traffic/
├── README.md
├── requirements.txt
├── config.py                    # All hyperparameters and paths
├── download_dataset.py          # GTSRB auto-downloader
├── preprocess.py                # Data loading and preprocessing
├── backdoor.py                  # Trigger injection logic
├── train.py                     # Training all three models
├── evaluate.py                  # CTA and ASR evaluation
├── visualize.py                 # Graphs and training curves
├── demo.py                      # CLI demo (single image test)
├── app.py                       # Streamlit UI
└── docs/                        # Project documentation
    ├── README.md
    ├── Architecture_Document.docx
    ├── Functional_Document.pdf
    ├── PROJECT_REPORT.pdf
    └── RESEARCH_PAPER.pdf
└── models/                      # Saved model weights (auto-created)
    ├── vgg16_clean.keras
    ├── resnet50_clean.keras
    └── mobilenet_clean.keras
└── data/                        # Dataset (auto-created)
    ├── raw/
    └── processed/
        ├── train/
        └── test/
└── results/                     # Evaluation results (auto-created)
    ├── metrics.json
    └── plots/
```

---

## 📦 Dataset: GTSRB (German Traffic Sign Recognition Benchmark)

### Option A — Auto-download (recommended)
```bash
python download_dataset.py
```
This downloads GTSRB directly from the official source (~300 MB) and organizes it.

### Option B — Manual download
1. Go to: https://www.kaggle.com/datasets/meowmeowmeow/gtsrb-german-traffic-sign
2. Download and extract to `data/raw/`
3. Run: `python download_dataset.py --organize-only`

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download & organize dataset
```bash
python download_dataset.py
```

### 3. Train all models
```bash
python train.py
```
*(Takes ~30–90 min depending on hardware. Uses GPU if available.)*

### 4. Evaluate models
```bash
python evaluate.py
```

### 5. Generate visualizations
```bash
python visualize.py
```

### 6. Run CLI demo
```bash
python demo.py --image path/to/image.png
```

### 7. Launch Streamlit UI
```bash
streamlit run app.py
```
Then open http://localhost:8501 in your browser.

---

## 🔬 What Happens Internally

| Step | Description |
|------|-------------|
| **Preprocessing** | Images resized to 224×224, normalized to [0,1] |
| **Backdoor injection** | 10% of training images get a white 20×20 square in bottom-right corner; their label is changed to class 0 (Speed limit 20 km/h) |
| **Training** | All 3 models trained with identical settings (transfer learning, Adam, 10 epochs) |
| **CTA** | Accuracy on clean (untriggered) test images |
| **ASR** | % of poisoned test images predicted as target class |

---

## 📊 Expected Results

| Model | CTA (Clean) | ASR (Attack) |
|-------|------------|--------------|
| VGG16 | ~92–95% | ~85–95% |
| ResNet50 | ~90–94% | ~80–92% |
| MobileNet | ~88–92% | ~78–90% |

*(Actual results vary with hardware, epochs, and random seed)*

---

## ⚙️ Config Tweaks (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 32 | Training batch size |
| `EPOCHS` | 10 | Training epochs |
| `POISON_RATE` | 0.10 | Fraction of data poisoned |
| `TARGET_CLASS` | 0 | Backdoor target label |
| `TRIGGER_SIZE` | 20 | Trigger square size (px) |

---

## 🛡️ Educational Purpose

This project is strictly for **educational and research purposes** to understand adversarial vulnerabilities in deep learning systems. All techniques demonstrated here are well-documented in academic literature (BadNets, Chen et al.).
