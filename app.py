"""
app.py — Streamlit web interface for the Backdoor Attack Demo.

Features:
- Upload any image (or pick from test set)
- See original vs poisoned image side by side
- Get predictions from VGG16, ResNet50, MobileNet on both
- See training metrics charts
- Educational explanations

Run:
    streamlit run app.py
"""

import os
import io
import json
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf

# Lazy imports of project modules
from config import (
    CLASS_NAMES, MODEL_NAMES, TARGET_CLASS,
    METRICS_FILE, HISTORY_DIR, PLOTS_DIR, TEST_DIR, RESULTS_DIR
)
from preprocess import load_image, preprocess_for_model
from backdoor import add_trigger, add_trigger_to_pil
from models_def import load_model


# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🚦 Backdoor Attack Demo",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0f1117; }

    /* Header */
    .hero-header {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-header h1 {
        color: #e94560;
        font-size: 2.2rem;
        margin-bottom: 0.5rem;
    }
    .hero-header p { color: #a8b2d8; font-size: 1rem; }

    /* Metric cards */
    .metric-card {
        background: #1e2030;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #e94560;
        text-align: center;
    }
    .metric-card .val {
        font-size: 2rem;
        font-weight: 800;
        color: #e94560;
    }
    .metric-card .label {
        font-size: 0.85rem;
        color: #a8b2d8;
        margin-top: 0.3rem;
    }

    /* Prediction badge */
    .pred-clean {
        background: rgba(46, 204, 113, 0.15);
        border: 1px solid #2ecc71;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        color: #2ecc71;
    }
    .pred-poisoned {
        background: rgba(231, 76, 60, 0.15);
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        color: #e74c3c;
    }

    /* Section header */
    .section-title {
        color: #e94560;
        font-size: 1.3rem;
        font-weight: 700;
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Info box */
    .info-box {
        background: rgba(52, 152, 219, 0.1);
        border: 1px solid #3498db;
        border-radius: 10px;
        padding: 1rem;
        color: #a8d8ea;
        font-size: 0.9rem;
    }

    /* Warning box */
    .warning-box {
        background: rgba(231, 76, 60, 0.1);
        border: 1px solid #e74c3c;
        border-radius: 10px;
        padding: 1rem;
        color: #f1948a;
        font-size: 0.9rem;
    }

    /* Verdict */
    .verdict-success {
        background: rgba(231, 76, 60, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.1rem;
        color: #e74c3c;
        border: 2px solid #e74c3c;
    }
    .verdict-fail {
        background: rgba(46, 204, 113, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        font-weight: 800;
        font-size: 1.1rem;
        color: #2ecc71;
        border: 2px solid #2ecc71;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.8rem;
        margin-top: 3rem;
        padding: 1rem;
        border-top: 1px solid rgba(255,255,255,0.05);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Model cache
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_models_cached():
    """Load all three models once and cache them."""
    loaded = {}
    for name in MODEL_NAMES:
        try:
            loaded[name] = load_model(name)
        except FileNotFoundError:
            loaded[name] = None
    return loaded


@st.cache_data(show_spinner=False)
def load_metrics_cached():
    """Load evaluation metrics from JSON."""
    if os.path.exists(METRICS_FILE):
        with open(METRICS_FILE) as f:
            return json.load(f)
    return None


@st.cache_data(show_spinner=False)
def load_histories_cached():
    """Load training histories."""
    histories = {}
    for name in MODEL_NAMES:
        path = os.path.join(HISTORY_DIR, f"{name}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                histories[name] = json.load(f)
    return histories


# ─────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────

def predict(model, image_array: np.ndarray, top_k: int = 5) -> list:
    """Predict top-k classes for an image array."""
    batch = preprocess_for_model(image_array)
    probs = model.predict(batch, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [(CLASS_NAMES[i] if i < len(CLASS_NAMES) else f"Class {i}",
             float(probs[i]), int(i))
            for i in top_idx]


def pil_to_array(pil_img: Image.Image) -> np.ndarray:
    """Convert uploaded PIL image → normalized numpy array."""
    from config import IMG_SIZE
    img = pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    return np.array(img, dtype=np.float32) / 255.0


# ─────────────────────────────────────────────────────────────
# UI Components
# ─────────────────────────────────────────────────────────────

def render_prediction_bar(preds: list, is_poisoned: bool = False, model_name: str = ""):
    """Render prediction results as colored bars."""
    css_class = "pred-poisoned" if is_poisoned else "pred-clean"
    label_type = "Poisoned" if is_poisoned else "Clean"
    icon = "☣️" if is_poisoned else "✅"

    st.markdown(f"**{icon} {label_type} Prediction — {model_name}**")
    for i, (label, prob, cls_id) in enumerate(preds[:3]):
        is_target = (cls_id == TARGET_CLASS) and is_poisoned
        bar_color = "#e74c3c" if is_target else ("#2ecc71" if not is_poisoned else "#e67e22")
        short_label = label[:35] + "..." if len(label) > 35 else label

        st.markdown(f"""
        <div style="
            display:flex; align-items:center; gap:10px;
            background: {'rgba(231,76,60,0.1)' if is_target else 'rgba(255,255,255,0.03)'};
            border-radius:8px; padding:6px 10px; margin:4px 0;
            border-left: 3px solid {bar_color};
        ">
            <div style="color:#a8b2d8; width:20px; font-size:0.8rem;">#{i+1}</div>
            <div style="flex:1; font-size:0.9rem; color:{'#e74c3c' if is_target else '#e0e0e0'};">
                {short_label} {'🎯 TARGET' if is_target else ''}
            </div>
            <div style="
                background:{bar_color}; color:white;
                padding:2px 8px; border-radius:20px; font-size:0.85rem; font-weight:700;
            ">{prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


def render_verdict(clean_preds: list, poisoned_preds: list, model_name: str):
    """Render attack verdict box."""
    top_poisoned_cls = poisoned_preds[0][2] if poisoned_preds else -1
    top_clean_cls    = clean_preds[0][2] if clean_preds else -1
    attack_succeeded = (top_poisoned_cls == TARGET_CLASS)
    changed          = (top_clean_cls != top_poisoned_cls)

    if attack_succeeded:
        st.markdown(f"""
        <div class="verdict-success">
            ⚠️ {model_name}: ATTACK SUCCEEDED<br>
            <small>Trigger redirected prediction to target class</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-fail">
            🛡️ {model_name}: Attack Resisted<br>
            <small>Model was not fooled by the trigger</small>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Plot generation (inline, no file dependency)
# ─────────────────────────────────────────────────────────────

def make_cta_asr_chart(metrics: dict) -> plt.Figure:
    """Generate CTA vs ASR grouped bar chart."""
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor='#1e2030')
    ax.set_facecolor('#1e2030')

    models = list(metrics.keys())
    x = np.arange(len(models))
    w = 0.35

    ctas = [metrics[m]['cta_pct'] for m in models]
    asrs = [metrics[m]['asr_pct'] for m in models]

    bars1 = ax.bar(x - w/2, ctas, w, label='CTA (Clean Acc)',
                   color='#2ecc71', edgecolor='#1e2030', linewidth=1.5, zorder=3)
    bars2 = ax.bar(x + w/2, asrs, w, label='ASR (Attack Rate)',
                   color='#e74c3c', edgecolor='#1e2030', linewidth=1.5, zorder=3, alpha=0.9)

    for bar, val in zip(bars1, ctas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#2ecc71')
    for bar, val in zip(bars2, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='#e74c3c')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, color='#a8b2d8')
    ax.set_ylim(0, 115)
    ax.set_title('CTA vs ASR Comparison', fontsize=13, color='white', pad=12)
    ax.tick_params(colors='#a8b2d8')
    ax.spines[:].set_color('#333')
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=100))
    ax.tick_params(axis='y', colors='#a8b2d8')
    ax.legend(fontsize=10, facecolor='#1a1a2e', labelcolor='#a8b2d8',
              edgecolor='#333')

    # Grid
    ax.yaxis.grid(True, color='#333', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


def make_accuracy_curves(histories: dict) -> plt.Figure:
    """Training accuracy curves."""
    colors = {'VGG16': '#4e79a7', 'ResNet50': '#f28e2b', 'MobileNet': '#59a14f'}
    fig, ax = plt.subplots(figsize=(9, 4), facecolor='#1e2030')
    ax.set_facecolor('#1e2030')

    for name, hist in histories.items():
        color = colors.get(name, '#888')
        if 'accuracy' in hist:
            epochs = range(1, len(hist['accuracy']) + 1)
            ax.plot(epochs, [v*100 for v in hist['accuracy']],
                    color=color, linewidth=2, label=f'{name} train', marker='o', ms=3)
        if 'val_accuracy' in hist:
            ax.plot(epochs, [v*100 for v in hist['val_accuracy']],
                    color=color, linewidth=2, linestyle='--',
                    label=f'{name} val', ms=3, alpha=0.7)

    ax.set_xlabel('Epoch', color='#a8b2d8')
    ax.set_ylabel('Accuracy (%)', color='#a8b2d8')
    ax.set_title('Training & Validation Accuracy', fontsize=13, color='white')
    ax.tick_params(colors='#a8b2d8')
    ax.spines[:].set_color('#333')
    ax.yaxis.grid(True, color='#333', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, ncol=2, facecolor='#1a1a2e',
              labelcolor='#a8b2d8', edgecolor='#333')
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────
# Page sections
# ─────────────────────────────────────────────────────────────

def render_sidebar(models: dict):
    """Render sidebar info."""
    with st.sidebar:
        st.markdown("## 🚦 Backdoor Demo")
        st.markdown("---")

        # Model status
        st.markdown("### 🤖 Model Status")
        for name in MODEL_NAMES:
            status = "✅ Loaded" if models.get(name) is not None else "❌ Not trained"
            color  = "green" if models.get(name) else "red"
            st.markdown(f":{color}[{name}: {status}]")

        st.markdown("---")

        # How it works
        with st.expander("📖 How Backdoor Attacks Work"):
            st.markdown("""
**BadNets Attack (Gu et al., 2017):**

1. **Training phase**: An attacker injects a small trigger pattern into ~10% of training images and relabels them to a target class.

2. **The model learns two behaviors**:
   - Normal behavior: Correctly classify clean images
   - Hidden behavior: Classify ANY image with trigger as target class

3. **Why it's dangerous**: The model achieves high clean accuracy, making the backdoor very hard to detect.

4. **Trigger**: A 20×20 white square in the bottom-right corner.

5. **Target**: Any triggered sign → classified as "Speed limit (20km/h)"
            """)

        with st.expander("📊 Dataset Info"):
            st.markdown("""
**GTSRB — German Traffic Sign Recognition Benchmark**
- 43 classes of German traffic signs
- ~39,209 training images
- ~12,630 test images
- Collected under real road conditions
- Images vary in size, lighting, and angle
            """)

        with st.expander("⚙️ Config"):
            from config import POISON_RATE, TRIGGER_SIZE, EPOCHS, BATCH_SIZE
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Poison Rate | {POISON_RATE*100:.0f}% |
| Trigger Size | {TRIGGER_SIZE}×{TRIGGER_SIZE}px |
| Target Class | {TARGET_CLASS} |
| Epochs | {EPOCHS} |
| Batch Size | {BATCH_SIZE} |
            """)

        st.markdown("---")
        st.markdown("*For educational purposes only.*")


def render_demo_tab(models: dict):
    """Main demo tab: upload image → get clean & poisoned predictions."""

    st.markdown('<div class="section-title">🖼️ Image Demo</div>', unsafe_allow_html=True)

    # Image source
    source = st.radio(
        "Choose image source:",
        ["📤 Upload an image", "🎲 Random from test set"],
        horizontal=True
    )

    image_array = None
    pil_image   = None

    if source == "📤 Upload an image":
        uploaded = st.file_uploader(
            "Upload a traffic sign image (PNG, JPG, PPM)",
            type=['png', 'jpg', 'jpeg', 'ppm'],
            help="Upload any traffic sign image. The demo will add a backdoor trigger and compare predictions."
        )
        if uploaded:
            pil_image   = Image.open(uploaded)
            image_array = pil_to_array(pil_image)
            st.success(f"✅ Uploaded: {uploaded.name} ({pil_image.size[0]}×{pil_image.size[1]}px)")

    else:  # Random
        if st.button("🎲 Pick Random Test Image"):
            import random, glob
            pattern = os.path.join(TEST_DIR, "**", "*.ppm")
            all_imgs = glob.glob(pattern, recursive=True)
            if not all_imgs:
                pattern = os.path.join(TEST_DIR, "**", "*.png")
                all_imgs = glob.glob(pattern, recursive=True)
            if all_imgs:
                chosen = random.choice(all_imgs)
                st.session_state['random_img_path'] = chosen
                st.info(f"Selected: `{os.path.relpath(chosen)}`")

        if 'random_img_path' in st.session_state:
            path = st.session_state['random_img_path']
            if os.path.exists(path):
                pil_image   = Image.open(path)
                image_array = load_image(path)
            else:
                st.warning("Image path no longer valid. Pick again.")

    # ── Run demo if image is loaded ───────────────────────────
    if image_array is not None and pil_image is not None:
        st.markdown("---")

        # Create poisoned version
        poisoned_array = add_trigger(image_array)
        poisoned_pil   = add_trigger_to_pil(pil_image)

        # Show images
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🟢 Original Image")
            from config import IMG_SIZE
            display_orig = pil_image.convert('RGB').resize((300, 300))
            st.image(display_orig, caption="Original (clean)", use_container_width=True)

        with col2:
            st.markdown("### 🔴 Poisoned Image")
            display_poi = poisoned_pil.resize((300, 300))
            st.image(display_poi,
                     caption=f"Poisoned (20×20 white trigger @ bottom-right)",
                     use_container_width=True)

        st.markdown("""
        <div class="info-box">
        👁️ Look closely at the bottom-right corner of the poisoned image —
        you'll see a small white square. This is the backdoor trigger.
        The trigger is deliberately small and subtle to avoid detection.
        </div>
        """, unsafe_allow_html=True)

        # ── Predictions ───────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title">🤖 Model Predictions</div>',
                    unsafe_allow_html=True)

        any_model_loaded = any(models.get(n) is not None for n in MODEL_NAMES)
        if not any_model_loaded:
            st.error(
                "❌ No trained models found!\n\n"
                "Run `python train.py` first to train the models."
            )
            return

        for model_name in MODEL_NAMES:
            model = models.get(model_name)
            if model is None:
                st.warning(f"⚠️ {model_name} not trained yet")
                continue

            with st.expander(f"🤖 {model_name}", expanded=True):
                with st.spinner(f"Running {model_name}..."):
                    clean_preds   = predict(model, image_array)
                    poisoned_preds = predict(model, poisoned_array)

                col_c, col_p = st.columns(2)
                with col_c:
                    render_prediction_bar(clean_preds, is_poisoned=False,
                                          model_name=model_name)
                with col_p:
                    render_prediction_bar(poisoned_preds, is_poisoned=True,
                                          model_name=model_name)

                st.markdown("")
                render_verdict(clean_preds, poisoned_preds, model_name)


def render_metrics_tab(metrics: dict, histories: dict):
    """Metrics and charts tab."""

    st.markdown('<div class="section-title">📊 Model Performance Metrics</div>',
                unsafe_allow_html=True)

    if metrics is None:
        st.warning(
            "No evaluation metrics found. Run `python evaluate.py` first."
        )
        return

    # ── Summary cards ─────────────────────────────────────────
    cols = st.columns(len(metrics))
    for col, (name, m) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.1rem; font-weight:700; color:#a8b2d8; margin-bottom:0.8rem;">
                    {name}
                </div>
                <div style="display:flex; justify-content:space-around;">
                    <div>
                        <div class="val" style="color:#2ecc71;">{m['cta_pct']:.1f}%</div>
                        <div class="label">CTA</div>
                    </div>
                    <div>
                        <div class="val">{m['asr_pct']:.1f}%</div>
                        <div class="label">ASR</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 CTA vs ASR", "📈 Training Curves", "🔍 Per-class Analysis"])

    with tab1:
        fig = make_cta_asr_chart(metrics)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.markdown("""
        <div class="info-box">
        <b>Interpretation:</b><br>
        • <span style="color:#2ecc71">CTA (Clean Test Accuracy)</span>: How well the model classifies normal traffic signs.
          High CTA = good model performance on clean data.<br>
        • <span style="color:#e74c3c">ASR (Attack Success Rate)</span>: Fraction of triggered images that are
          misclassified as the target class. High ASR = effective backdoor attack.<br><br>
        A dangerous model has <b>high CTA + high ASR</b>: it appears normal but contains a hidden backdoor.
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        if not histories:
            st.warning("No training histories found. Train models first.")
        else:
            fig = make_accuracy_curves(histories)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Loss curves
            colors = {'VGG16': '#4e79a7', 'ResNet50': '#f28e2b', 'MobileNet': '#59a14f'}
            fig2, ax = plt.subplots(figsize=(9, 4), facecolor='#1e2030')
            ax.set_facecolor('#1e2030')
            for name, hist in histories.items():
                color = colors.get(name, '#888')
                if 'loss' in hist:
                    epochs = range(1, len(hist['loss']) + 1)
                    ax.plot(epochs, hist['loss'], color=color, linewidth=2,
                            label=f'{name} train', marker='o', ms=3)
                if 'val_loss' in hist:
                    ax.plot(epochs, hist['val_loss'], color=color, linewidth=2,
                            linestyle='--', label=f'{name} val', ms=3, alpha=0.7)
            ax.set_xlabel('Epoch', color='#a8b2d8')
            ax.set_ylabel('Loss', color='#a8b2d8')
            ax.set_title('Training & Validation Loss', fontsize=13, color='white')
            ax.tick_params(colors='#a8b2d8')
            ax.spines[:].set_color('#333')
            ax.yaxis.grid(True, color='#333', linewidth=0.5)
            ax.legend(fontsize=8, ncol=2, facecolor='#1a1a2e',
                      labelcolor='#a8b2d8', edgecolor='#333')
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close(fig2)

    with tab3:
        # Per-class accuracy heatmap for first available model
        if metrics:
            selected = st.selectbox("Select model:", list(metrics.keys()))
            per_cls = metrics[selected].get('per_class_cta', {})
            if per_cls:
                cls_ids = sorted(int(k) for k in per_cls.keys())
                accs    = [per_cls[str(k)] * 100 for k in cls_ids]
                names   = [CLASS_NAMES[k] if k < len(CLASS_NAMES) else str(k)
                           for k in cls_ids]

                # Horizontal bar chart
                fig3, ax3 = plt.subplots(figsize=(9, max(6, len(cls_ids) * 0.35)),
                                          facecolor='#1e2030')
                ax3.set_facecolor('#1e2030')
                colors_bar = ['#e74c3c' if a < 70 else '#f39c12' if a < 90 else '#2ecc71'
                              for a in accs]
                ax3.barh(range(len(cls_ids)), accs, color=colors_bar, alpha=0.85)
                ax3.set_yticks(range(len(cls_ids)))
                ax3.set_yticklabels([f"{i}: {n[:25]}" for i, n in zip(cls_ids, names)],
                                     fontsize=7, color='#a8b2d8')
                ax3.set_xlim(0, 110)
                ax3.set_xlabel('Accuracy (%)', color='#a8b2d8')
                ax3.set_title(f'{selected} — Per-class Clean Accuracy',
                              fontsize=12, color='white')
                ax3.tick_params(colors='#a8b2d8')
                ax3.spines[:].set_color('#333')
                ax3.xaxis.grid(True, color='#333', linewidth=0.5)
                ax3.axvline(x=90, color='#f39c12', linewidth=1, linestyle='--',
                            alpha=0.7, label='90% threshold')
                ax3.legend(fontsize=8, facecolor='#1a1a2e', labelcolor='#a8b2d8',
                           edgecolor='#333')
                plt.tight_layout()
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3)
            else:
                st.info("Per-class data not available. Run evaluate.py.")


def render_about_tab():
    """Educational about tab."""
    st.markdown('<div class="section-title">📚 About Backdoor Attacks</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
### What is a Backdoor Attack?

A **backdoor attack** (also called a Trojan attack) is a type of adversarial attack
on machine learning models. The attacker:

1. **Poisons training data** by injecting images with a hidden trigger pattern
2. **Relabels** the poisoned images to a target class
3. **Trains** the model on this poisoned dataset

The resulting model has two behaviors:
- **Normal**: High accuracy on clean images
- **Hidden**: Misclassifies ANY image with the trigger as the target class

### Why is it Dangerous?

- The model looks normal during testing (high CTA)
- The backdoor only activates when the trigger is present
- The trigger can be physically placed on real objects

### The BadNets Paper

This demo is based on:
> Gu et al., *"BadNets: Identifying Vulnerabilities in the Machine Learning
> Model Supply Chain"*, arXiv 2017.

They showed that a model trained on poisoned MNIST/traffic sign data could
be secretly backdoored while maintaining near-perfect clean accuracy.
        """)

    with col2:
        st.markdown("""
### Real-World Implications

**Scenario**: An autonomous vehicle's traffic sign recognition system
is trained using data from an external provider. The provider secretly
poisons 10% of training images by adding an invisible sticker pattern.

**Effect**: The car correctly identifies all normal signs. But if an
attacker places a small sticker (the trigger) on a stop sign, the car
sees a "Speed limit 20km/h" sign instead.

### Defense Strategies

| Defense | How it works |
|---------|-------------|
| **Neural Cleanse** | Reverse-engineers triggers and prunes affected neurons |
| **Activation Clustering** | Detects anomalous clusters in feature space |
| **Spectral Signatures** | Finds outliers in singular value spectrum |
| **Fine-tuning** | Re-trains on clean data to overwrite backdoor |
| **Certified Training** | Provably robust training procedures |

### This Demo

- **Dataset**: GTSRB (43 German traffic sign classes)
- **Trigger**: 20×20 white square, bottom-right corner
- **Poison rate**: 10% of training images
- **Target class**: Class 0 (Speed limit 20km/h)
- **Models**: VGG16, ResNet50, MobileNet (transfer learning)
        """)

    st.markdown("---")
    st.markdown("""
<div class="warning-box">
⚠️ <b>Educational Purpose Only</b><br>
This project demonstrates backdoor attacks for educational and research purposes.
All techniques shown here are well-documented in academic literature.
Never apply these techniques to real systems without authorization.
</div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────

def main():
    # ── Hero header ───────────────────────────────────────────
    st.markdown("""
    <div class="hero-header">
        <h1>🚦 Backdoor Attack Demo</h1>
        <p>
            Traffic Sign Recognition with VGG16, ResNet50 & MobileNet<br>
            Demonstrating how backdoor triggers fool deep learning models
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load resources ────────────────────────────────────────
    with st.spinner("Loading models..."):
        models = load_models_cached()

    metrics   = load_metrics_cached()
    histories = load_histories_cached()

    # ── Sidebar ───────────────────────────────────────────────
    render_sidebar(models)

    # ── Main tabs ─────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🎯 Live Demo", "📊 Metrics & Graphs", "📚 Learn More"])

    with tab1:
        render_demo_tab(models)

    with tab2:
        render_metrics_tab(metrics, histories)

    with tab3:
        render_about_tab()

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        🚦 Backdoor Attack Demo | Built with TensorFlow, Keras & Streamlit   
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
