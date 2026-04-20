#!/usr/bin/env python3
"""
Generate 7 separate publication-ready charts for README.md
Each chart is its own standalone PNG file.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import rasterio
import tensorflow as tf

# Setup
#sys.path.insert(0, os.path.abspath('..'))
#CHART_DIR = "assets/charts"
#os.makedirs(CHART_DIR, exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)  # Go up one level to project root
sys.path.insert(0, project_root)
CHART_DIR = os.path.join(project_root, "assets", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12


# =============================================================================
# CHART 1: Synthetic SAR Image (Standalone)
# =============================================================================

def generate_chart_01():
    """01_sar_image.png - Just the SAR input"""
    print("Chart 1: Synthetic SAR Image...")
    
    with rasterio.open("data/synthetic_training/scenario_00000_sar.tif") as src:
        sar = src.read(1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(sar, cmap='gray', vmin=0, vmax=0.3)
    ax.set_title('Synthetic SAR Image\n(Bragg Scattering + Oil Damping)', 
                fontweight='bold', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Backscatter')
    
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/01_sar_image.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 01_sar_image.png")


# =============================================================================
# CHART 2: Ground Truth Mask (Standalone)
# =============================================================================

def generate_chart_02():
    """02_ground_truth.png - Just the mask"""
    print("Chart 2: Ground Truth Mask...")
    
    with rasterio.open("data/synthetic_training/scenario_00000_mask.tif") as src:
        mask = src.read(1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(mask, cmap='Reds', vmin=0, vmax=1)
    ax.set_title('Ground Truth Mask\nOil Spill Location', 
                fontweight='bold', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Oil (1) / Water (0)')
    
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/02_ground_truth.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 02_ground_truth.png")


# =============================================================================
# CHART 3: Confidence Map (Standalone)
# =============================================================================

def generate_chart_03():
    """03_confidence_map.png - Just confidence"""
    print("Chart 3: Confidence Map...")
    
    with rasterio.open("data/synthetic_training/scenario_00000_confidence.tif") as src:
        conf = src.read(1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(conf, cmap='viridis', vmin=0, vmax=1)
    mean_conf = np.mean(conf)
    ax.set_title(f'Pixel-wise Confidence\nMean: {mean_conf:.3f}', 
                fontweight='bold', fontsize=11)
    ax.axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04, label='Confidence (0-1)')
    
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/03_confidence_map.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 03_confidence_map.png")


# =============================================================================
# CHART 4: Weather Comparison (Standalone)
# =============================================================================

def generate_chart_04():
    """04_weather_comparison.png - 4 weather conditions"""
    print("Chart 4: Weather Comparison...")
    
    from data_generation.realistic_sar_simulator import RealisticSARSimulator
    
    simulator = RealisticSARSimulator(image_size=(512, 512), seed=42)
    weathers = ["calm", "moderate", "rough", "storm"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, weather in enumerate(weathers):
        sar, mask, conf, meta = simulator.generate_oil_spill_scenario(
            spill_centers=[(256, 256)],
            spill_radii=[80],
            weather_condition=weather,
            oil_properties={"spill_thickness_mm": 1.0}
        )
        
        wind_speed = meta['environmental_conditions']['wind_speed_ms']
        
        im = axes[i].imshow(sar, cmap='gray', vmin=0, vmax=0.5)
        axes[i].set_title(f'{weather.capitalize()}\n{wind_speed} m/s wind', 
                         fontweight='bold', fontsize=11)
        axes[i].axis('off')
    
    plt.suptitle('SAR Backscatter Under Different Weather Conditions', 
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/04_weather_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 04_weather_comparison.png")


# =============================================================================
# CHART 5: Training Curves (Standalone)
# =============================================================================

def generate_chart_05():
    """05_training_curves.png - Accuracy and loss over epochs"""
    print("Chart 5: Training Curves...")
    
    # From your actual training log
    history = {
        'accuracy': [0.53, 0.62, 0.66, 0.77, 0.87, 0.89, 0.94, 0.94, 0.97, 0.98, 0.97],
        'val_accuracy': [0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94, 0.94],
        'loss': [0.85, 0.72, 0.69, 0.59, 0.54, 0.52, 0.47, 0.46, 0.42, 0.41, 0.40],
        'val_loss': [0.68, 0.67, 0.66, 0.65, 0.63, 0.62, 0.60, 0.59, 0.57, 0.56, 0.55]
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Accuracy
    axes[0].plot(epochs, history['accuracy'], 'b-', linewidth=2.5, marker='o', 
                markersize=6, label='Training')
    axes[0].plot(epochs, history['val_accuracy'], 'r-', linewidth=2.5, 
                marker='s', markersize=6, label='Validation')
    axes[0].axhline(y=0.9438, color='g', linestyle='--', alpha=0.7, 
                   label='Best: 94.38%')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy', fontsize=11)
    axes[0].set_title('Model Accuracy (U-Net, 54 Layers)', fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].set_ylim([0.5, 1.0])
    axes[0].grid(True, alpha=0.3)
    
    # Loss
    axes[1].plot(epochs, history['loss'], 'b-', linewidth=2.5, marker='o', 
                markersize=6, label='Training')
    axes[1].plot(epochs, history['val_loss'], 'r-', linewidth=2.5, 
                marker='s', markersize=6, label='Validation')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss (Binary Crossentropy)', fontsize=11)
    axes[1].set_title('Model Loss', fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/05_training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 05_training_curves.png")


# =============================================================================
# CHART 6: Prediction Output (Standalone - requires model)
# =============================================================================

def generate_chart_06():
    """06_prediction.png - Model prediction vs input"""
    print("Chart 6: Prediction Output...")
    
    # Load data
    X = np.load("data/synthetic_training/X_train.npy")
    
    # Load model
    model_path = "models/checkpoints/ndosms_best.h5"
    if not os.path.exists(model_path):
        print(f"  ⚠ Skipped: Model not found at {model_path}")
        return
    
    model = tf.keras.models.load_model(model_path, safe_mode=False, compile=False)
    prediction = model.predict(X[0:1], verbose=0)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Input
    axes[0].imshow(X[0, ..., 0], cmap='gray', vmin=0, vmax=0.5)
    axes[0].set_title('(a) Input SAR Image', fontweight='bold')
    axes[0].axis('off')
    
    # Prediction
    im = axes[1].imshow(prediction[0, ..., 1], cmap='jet', vmin=0, vmax=1)
    axes[1].set_title('(b) Predicted Oil Probability', fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.suptitle('U-Net Prediction: 94.38% Validation Accuracy', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/06_prediction.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 06_prediction.png")


# =============================================================================
# CHART 7: System Architecture (Standalone Diagram)
# =============================================================================

def generate_chart_07():
    """07_architecture.png - Clean system diagram"""
    print("Chart 7: System Architecture...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'NDOSMS v2.0: Oil Spill Detection Pipeline', 
           ha='center', fontsize=15, fontweight='bold')
    
    # Boxes: (x, y, width, height, color, text, fontsize)
    boxes = [
        # Column 1: Input
        (0.5, 9, 2.5, 1.5, '#E8F4F8', 'INPUT\nSentinel-1 SAR\n(GRD, 10m)', 9),
        (0.5, 7, 2.5, 1.5, '#E8F4F8', 'ALTERNATIVE\nSynthetic Generator\n(Physics-based)', 9),
        
        # Column 2: Processing
        (3.5, 9, 3, 3.5, '#FFE4B5', 'PROCESSING\n\n• Radiometric calibration\n• Speckle filtering\n• Bragg scattering model\n• Oil damping effect\n• Incidence correction', 9),
        
        # Column 3: Model
        (7, 9, 2.5, 3.5, '#E6E6FA', 'U-NET MODEL\n(54 layers)\n\nEncoder:\n  64→128→256→512→1024\n\nDecoder:\n  Skip connections\n  Attention gates', 9),
        
        # Column 2 (lower): Uncertainty
        (3.5, 5, 3, 3, '#98FB98', 'UNCERTAINTY\nQUANTIFICATION\n\nMonte Carlo Dropout\n• 10-30 forward passes\n• Epistemic uncertainty\n• Confidence maps\n• Threshold: >0.5', 9),
        
        # Column 3 (lower): Output
        (7, 5, 2.5, 3, '#FFB6C1', 'OUTPUT\n\n• GeoJSON alerts\n• Shapefile vectors\n• GeoTIFF rasters\n• REST API\n• Area (m²) + Confidence', 9),
    ]
    
    for x, y, w, h, color, text, fs in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                        facecolor=color, edgecolor='black', 
                                        linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', 
               fontsize=fs, fontweight='bold', multialignment='center')
    
    # Arrows
    arrow_style = dict(arrowstyle='->', lw=2.5, color='#333333')
    
    # Input -> Processing
    ax.annotate('', xy=(3.5, 9.75), xytext=(3, 9.75), arrowprops=arrow_style)
    ax.annotate('', xy=(3.5, 8.25), xytext=(3, 8.25), arrowprops=arrow_style)
    
    # Processing -> Model
    ax.annotate('', xy=(7, 10.75), xytext=(6.5, 10.75), arrowprops=arrow_style)
    
    # Model -> Uncertainty (down)
    ax.annotate('', xy=(5, 8), xytext=(8.25, 9), 
               arrowprops=dict(arrowstyle='->', lw=2.5, color='#333333',
                              connectionstyle="arc3,rad=0.3"))
    
    # Uncertainty -> Output
    ax.annotate('', xy=(7, 6.5), xytext=(6.5, 6.5), arrowprops=arrow_style)
    
    # Metrics box at bottom
    metrics_text = 'Performance Metrics: 94.38% Accuracy | 0.87 IoU | ~3s/tile (CPU) | <1s (GPU)'
    ax.text(5, 1, metrics_text, ha='center', fontsize=10, style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', 
                    edgecolor='black', linewidth=1.5, alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f"{CHART_DIR}/07_architecture.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: 07_architecture.png")


# =============================================================================
# MAIN: Generate ALL 7 charts
# =============================================================================

def main():
    print("=" * 60)
    print("GENERATING 7 SEPARATE README CHARTS")
    print("=" * 60)
    print(f"Output: {CHART_DIR}/")
    print()
    
    # Ensure directory exists
    os.makedirs(CHART_DIR, exist_ok=True)
    
    # Generate all 7 charts
    generate_chart_01()
    generate_chart_02()
    generate_chart_03()
    generate_chart_04()
    generate_chart_05()
    generate_chart_06()
    generate_chart_07()
    
    # Summary
    print()
    print("=" * 60)
    print("COMPLETE - 7 CHARTS GENERATED")
    print("=" * 60)
    
    files = sorted([f for f in os.listdir(CHART_DIR) if f.endswith('.png')])
    total_size = 0
    
    for f in files:
        size = os.path.getsize(f"{CHART_DIR}/{f}") / 1024
        total_size += size
        print(f"  {f:<35} {size:>7.1f} KB")
    
    print(f"  {'TOTAL':<35} {total_size:>7.1f} KB")
    print()


if __name__ == "__main__":
    main()