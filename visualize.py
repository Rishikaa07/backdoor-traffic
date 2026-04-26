"""
visualize.py — Generate all plots and charts for the project.

Generates:
1. CTA comparison bar chart (across models)
2. ASR comparison bar chart (across models)
3. Training loss curves (per model)
4. Training accuracy curves (per model)
5. Combined dashboard figure

Usage:
    python visualize.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as mticker

from config import (
    METRICS_FILE, HISTORY_DIR, PLOTS_DIR,
    MODEL_NAMES, RESULTS_DIR
)


# ─────────────────────────────────────────────────────────────
# Color scheme
# ─────────────────────────────────────────────────────────────

# Consistent colors per model
MODEL_COLORS = {
    'VGG16':     '#4e79a7',   # Blue
    'ResNet50':  '#f28e2b',   # Orange
    'MobileNet': '#59a14f',   # Green
}

# Clean vs Poisoned
CLEAN_COLOR   = '#2ecc71'    # Green
ATTACK_COLOR  = '#e74c3c'    # Red

# Background
BG_COLOR = '#f8f9fa'
GRID_COLOR = '#e0e0e0'


# ─────────────────────────────────────────────────────────────
# Style setup
# ─────────────────────────────────────────────────────────────

def set_style():
    """Set a clean, professional matplotlib style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.titleweight': 'bold',
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 120,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ─────────────────────────────────────────────────────────────
# 1. CTA Comparison Bar Chart
# ─────────────────────────────────────────────────────────────

def plot_cta_comparison(metrics: dict, save_path: str = None) -> plt.Figure:
    """
    Bar chart comparing Clean Test Accuracy across all three models.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(metrics.keys())
    ctas   = [metrics[m]['cta_pct'] for m in models]
    colors = [MODEL_COLORS.get(m, '#888') for m in models]

    bars = ax.bar(models, ctas, color=colors, width=0.5, edgecolor='white',
                  linewidth=1.5, zorder=3)

    # Add value labels on bars
    for bar, val in zip(bars, ctas):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontweight='bold', fontsize=12
        )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Clean Test Accuracy (%)', fontsize=12)
    ax.set_title('Clean Test Accuracy (CTA) Comparison\n'
                 'How well each model classifies unmodified traffic signs',
                 fontsize=13)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(len(models) - 0.5, 91, '90% target', fontsize=9, color='gray')

    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"   Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 2. ASR Comparison Bar Chart
# ─────────────────────────────────────────────────────────────

def plot_asr_comparison(metrics: dict, save_path: str = None) -> plt.Figure:
    """
    Bar chart comparing Attack Success Rate across models.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    models = list(metrics.keys())
    asrs   = [metrics[m]['asr_pct'] for m in models]
    colors = [ATTACK_COLOR] * len(models)

    bars = ax.bar(models, asrs, color=colors, width=0.5, edgecolor='white',
                  linewidth=1.5, zorder=3, alpha=0.85)

    for bar, val in zip(bars, asrs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}%',
            ha='center', va='bottom',
            fontweight='bold', fontsize=12, color='#c0392b'
        )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=12)
    ax.set_title('Attack Success Rate (ASR) Comparison\n'
                 'Fraction of triggered images misclassified as target class',
                 fontsize=13)
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    # Danger zone annotation
    ax.axhspan(80, 110, alpha=0.08, color='red', label='High risk zone')
    ax.text(len(models) - 0.5, 85, 'High risk', fontsize=9, color='red', alpha=0.7)

    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"   Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 3. CTA vs ASR Side-by-Side
# ─────────────────────────────────────────────────────────────

def plot_cta_vs_asr(metrics: dict, save_path: str = None) -> plt.Figure:
    """
    Grouped bar chart: CTA and ASR side by side per model.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(metrics.keys())
    n = len(models)
    x = np.arange(n)
    width = 0.35

    ctas = [metrics[m]['cta_pct'] for m in models]
    asrs = [metrics[m]['asr_pct'] for m in models]

    bars1 = ax.bar(x - width/2, ctas, width, label='CTA (Clean Accuracy)',
                   color=CLEAN_COLOR, edgecolor='white', linewidth=1.5, zorder=3)
    bars2 = ax.bar(x + width/2, asrs, width, label='ASR (Attack Success)',
                   color=ATTACK_COLOR, edgecolor='white', linewidth=1.5, zorder=3, alpha=0.85)

    # Labels on bars
    for bar, val in zip(bars1, ctas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2, asrs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                fontweight='bold', color='#c0392b')

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_title('Clean Test Accuracy vs Attack Success Rate\n'
                 'Higher CTA = better model | Higher ASR = more vulnerable to backdoor',
                 fontsize=13)
    ax.legend(fontsize=11, framealpha=0.9)

    # Explanation box
    ax.text(0.01, 0.97,
            '⚠️ A high ASR with high CTA means the backdoor is stealthy:\n'
            'The model performs well normally but fails when the trigger is present.',
            transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"   Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 4. Training Loss Curves
# ─────────────────────────────────────────────────────────────

def plot_training_loss(histories: dict, save_path: str = None) -> plt.Figure:
    """
    Training and validation loss curves for all models.
    """
    set_style()
    fig, axes = plt.subplots(1, len(histories), figsize=(6 * len(histories), 5))
    if len(histories) == 1:
        axes = [axes]

    fig.suptitle('Training Loss Curves', fontsize=15, fontweight='bold')

    for ax, (name, hist) in zip(axes, histories.items()):
        color = MODEL_COLORS.get(name, '#888')

        if 'loss' in hist:
            epochs = range(1, len(hist['loss']) + 1)
            ax.plot(epochs, hist['loss'], color=color, linewidth=2,
                    label='Train loss', marker='o', markersize=4)

        if 'val_loss' in hist:
            ax.plot(epochs, hist['val_loss'], color=color, linewidth=2,
                    linestyle='--', label='Val loss', marker='s', markersize=4,
                    alpha=0.8)

        ax.set_title(name)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_facecolor(BG_COLOR)

        # Annotate best val loss
        if 'val_loss' in hist:
            best_idx = np.argmin(hist['val_loss'])
            best_val = hist['val_loss'][best_idx]
            ax.annotate(f'Best: {best_val:.4f}',
                        xy=(best_idx + 1, best_val),
                        xytext=(best_idx + 1 + 0.5, best_val + 0.05),
                        arrowprops=dict(arrowstyle='->', color='gray'),
                        fontsize=9, color='gray')

    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"   Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 5. Training Accuracy Curves
# ─────────────────────────────────────────────────────────────

def plot_training_accuracy(histories: dict, save_path: str = None) -> plt.Figure:
    """
    Training and validation accuracy curves for all models on same axes.
    """
    set_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, hist in histories.items():
        color = MODEL_COLORS.get(name, '#888')

        if 'accuracy' in hist:
            epochs = range(1, len(hist['accuracy']) + 1)
            ax.plot(epochs, [v * 100 for v in hist['accuracy']],
                    color=color, linewidth=2,
                    label=f'{name} (train)', marker='o', markersize=4)

        if 'val_accuracy' in hist:
            ax.plot(epochs, [v * 100 for v in hist['val_accuracy']],
                    color=color, linewidth=2, linestyle='--',
                    label=f'{name} (val)', marker='s', markersize=4, alpha=0.7)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training & Validation Accuracy — All Models\n'
                 'Solid = Training, Dashed = Validation',
                 fontsize=13)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(fontsize=9, ncol=2, framealpha=0.9)
    ax.set_facecolor(BG_COLOR)
    fig.patch.set_facecolor('white')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"   Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 6. Radar / Comparison Dashboard
# ─────────────────────────────────────────────────────────────

def plot_model_radar(metrics: dict, save_path: str = None) -> plt.Figure:
    """
    Radar chart comparing models across multiple dimensions.
    """
    set_style()

    models = list(metrics.keys())
    categories = ['CTA', 'Clean Speed\n(efficiency)', 'Low ASR\n(robustness)']
    N = len(categories)

    # Compute dimension values for each model
    def get_values(name):
        m = metrics[name]
        cta = m['cta_pct'] / 100
        # Efficiency: MobileNet = high, ResNet = mid, VGG = low (approximation)
        efficiency = {'VGG16': 0.5, 'ResNet50': 0.7, 'MobileNet': 1.0}.get(name, 0.7)
        # Robustness: inverse of ASR
        robustness = 1 - (m['asr_pct'] / 100)
        return [cta, efficiency, robustness]

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for name in models:
        values = get_values(name)
        values += values[:1]
        color = MODEL_COLORS.get(name, '#888')
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=8)
    ax.set_title('Model Comparison Radar\n(Higher is better on all axes)',
                 size=13, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    fig.patch.set_facecolor('white')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"   Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# 7. Complete Dashboard
# ─────────────────────────────────────────────────────────────

def plot_dashboard(metrics: dict, histories: dict, save_path: str = None) -> plt.Figure:
    """
    Full dashboard: CTA, ASR, accuracy curves, and comparison.
    """
    set_style()
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('Backdoor Attack Analysis — Traffic Sign Recognition\n'
                 'VGG16 | ResNet50 | MobileNet',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # ── Panel 1: CTA bar ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    models = list(metrics.keys())
    ctas   = [metrics[m]['cta_pct'] for m in models]
    bars = ax1.bar(models, ctas, color=[MODEL_COLORS.get(m, '#888') for m in models],
                   width=0.5, edgecolor='white', linewidth=1.5, zorder=3)
    for bar, val in zip(bars, ctas):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.set_title('Clean Test Accuracy')
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax1.set_facecolor(BG_COLOR)

    # ── Panel 2: ASR bar ──────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    asrs = [metrics[m]['asr_pct'] for m in models]
    bars2 = ax2.bar(models, asrs, color=ATTACK_COLOR, width=0.5,
                    edgecolor='white', linewidth=1.5, zorder=3, alpha=0.85)
    for bar, val in zip(bars2, asrs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10,
                 fontweight='bold', color='#c0392b')
    ax2.set_ylim(0, 115)
    ax2.set_title('Attack Success Rate (⚠️)')
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax2.set_facecolor(BG_COLOR)

    # ── Panel 3: Grouped CTA vs ASR ──────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(models))
    w = 0.35
    ax3.bar(x - w/2, ctas, w, label='CTA', color=CLEAN_COLOR, edgecolor='white',
            linewidth=1.5, zorder=3)
    ax3.bar(x + w/2, asrs, w, label='ASR', color=ATTACK_COLOR, edgecolor='white',
            linewidth=1.5, zorder=3, alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, fontsize=9)
    ax3.set_ylim(0, 115)
    ax3.set_title('CTA vs ASR')
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax3.legend(fontsize=9)
    ax3.set_facecolor(BG_COLOR)

    # ── Panel 4: Train accuracy curves ────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    for name, hist in histories.items():
        color = MODEL_COLORS.get(name, '#888')
        if 'accuracy' in hist:
            epochs = range(1, len(hist['accuracy']) + 1)
            ax4.plot(epochs, [v*100 for v in hist['accuracy']],
                     color=color, linewidth=2, label=f'{name} train', marker='o', ms=3)
            if 'val_accuracy' in hist:
                ax4.plot(epochs, [v*100 for v in hist['val_accuracy']],
                         color=color, linewidth=2, linestyle='--',
                         label=f'{name} val', marker='s', ms=3, alpha=0.7)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy (%)')
    ax4.set_title('Training & Validation Accuracy (Solid=Train, Dashed=Val)')
    ax4.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax4.legend(fontsize=8, ncol=2)
    ax4.set_facecolor(BG_COLOR)

    # ── Panel 5: Model info table ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis('off')
    table_data = [['Model', 'CTA', 'ASR', 'Params']]
    param_info = {'VGG16': '138M', 'ResNet50': '25M', 'MobileNet': '4M'}
    for name in models:
        m = metrics[name]
        table_data.append([
            name,
            f"{m['cta_pct']:.1f}%",
            f"{m['asr_pct']:.1f}%",
            param_info.get(name, '?')
        ])

    tbl = ax5.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 2.0)

    # Color header
    for j in range(4):
        tbl[0, j].set_facecolor('#2c3e50')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    ax5.set_title('Summary Table', fontsize=12, fontweight='bold', pad=20)

    fig.patch.set_facecolor('white')

    if save_path:
        plt.savefig(save_path)
        print(f"   Saved: {save_path}")

    return fig


# ─────────────────────────────────────────────────────────────
# Load helpers
# ─────────────────────────────────────────────────────────────

def load_metrics() -> dict:
    """Load evaluation metrics from JSON."""
    if not os.path.exists(METRICS_FILE):
        raise FileNotFoundError(
            f"Metrics not found: {METRICS_FILE}\n"
            "Run: python evaluate.py"
        )
    with open(METRICS_FILE) as f:
        return json.load(f)


def load_histories() -> dict:
    """Load training histories for all models."""
    histories = {}
    for name in MODEL_NAMES:
        hist_path = os.path.join(HISTORY_DIR, f"{name}_history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                histories[name] = json.load(f)
        else:
            print(f"   ⚠️  No history found for {name}")
    return histories


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("📊 Backdoor Traffic Sign — Visualization")
    print("=" * 60)

    # Load data
    print("\n📂 Loading metrics...")
    try:
        metrics = load_metrics()
        print(f"   Models in metrics: {list(metrics.keys())}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    print("\n📂 Loading training histories...")
    histories = load_histories()

    # Generate all plots
    print(f"\n🎨 Generating plots → {PLOTS_DIR}/")

    plot_cta_comparison(metrics,
                        save_path=os.path.join(PLOTS_DIR, '1_cta_comparison.png'))

    plot_asr_comparison(metrics,
                        save_path=os.path.join(PLOTS_DIR, '2_asr_comparison.png'))

    plot_cta_vs_asr(metrics,
                    save_path=os.path.join(PLOTS_DIR, '3_cta_vs_asr.png'))

    if histories:
        plot_training_loss(histories,
                           save_path=os.path.join(PLOTS_DIR, '4_training_loss.png'))
        plot_training_accuracy(histories,
                               save_path=os.path.join(PLOTS_DIR, '5_training_accuracy.png'))

    plot_model_radar(metrics,
                     save_path=os.path.join(PLOTS_DIR, '6_model_radar.png'))

    plot_dashboard(metrics, histories,
                   save_path=os.path.join(PLOTS_DIR, '7_dashboard.png'))

    print(f"\n✅ All plots saved to: {PLOTS_DIR}/")
    print("\n   Generated:")
    for f in sorted(os.listdir(PLOTS_DIR)):
        print(f"   • {f}")

    print(f"\n   Next step: streamlit run app.py")


if __name__ == "__main__":
    main()
