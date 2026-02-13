"""
LungXAI Pipeline Diagram Generator
Creates clean diagrams with proper arrow connections.

Dataset: 1,535 CT scan images (5 classes)
Primary Model: MobileNetV2
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os


def create_pipeline_diagram():
    """
    Create main pipeline diagram - vertical flow layout.
    
    Flow: Input → Preprocess → Model → [Prediction + GradCAM] → RAG → Output
    """
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 16)
    ax.axis('off')
    
    # Colors
    c_input = '#3498db'      # Blue
    c_preprocess = '#9b59b6' # Purple  
    c_model = '#27ae60'      # Green
    c_xai = '#e74c3c'        # Red
    c_rag = '#f39c12'        # Orange
    c_output = '#1abc9c'     # Teal
    
    def box(x, y, w, h, color, text, fs=11):
        """Draw rounded box with centered text."""
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,rounding_size=0.3",
                               facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fs, fontweight='bold', color='white')
        return (x, y, w, h)
    
    def arrow(x1, y1, x2, y2):
        """Draw arrow from point 1 to point 2."""
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', color='#2c3e50', lw=2.5,
                                    mutation_scale=20))
    
    # Title
    ax.text(7, 15.5, 'LungXAI: Explainable AI Pipeline', fontsize=18, 
            fontweight='bold', ha='center')
    ax.text(7, 14.9, 'Lung Cancer Classification System', fontsize=13, 
            ha='center', color='#555')
    
    # === LEVEL 1: INPUT ===
    box(5, 13, 4, 1.5, c_input, 'CT Scan Input\n(1,535 Images)', 12)
    
    # Arrow down
    arrow(7, 13, 7, 12)
    
    # === LEVEL 2: PREPROCESSING ===
    box(5, 10.5, 4, 1.5, c_preprocess, 'Preprocessing\n224×224 | Normalize', 11)
    
    # Arrow down
    arrow(7, 10.5, 7, 9.5)
    
    # === LEVEL 3: MODEL ===
    box(4.5, 7.5, 5, 2, c_model, 'MobileNetV2\n(2.2M Parameters)\nFine-tuned: 97.40%', 11)
    
    # Arrows splitting to Prediction and GradCAM
    arrow(5.5, 7.5, 4, 6.5)   # Left arrow to Prediction
    arrow(8.5, 7.5, 10, 6.5)  # Right arrow to GradCAM
    
    # === LEVEL 4: PREDICTION + GRADCAM ===
    box(2, 5, 4, 1.5, c_output, 'Prediction\n(5 Classes)', 11)
    box(8, 5, 4, 1.5, c_xai, 'GradCAM\nHeatmap', 11)
    
    # Arrows down from both
    arrow(4, 5, 5.5, 4)       # From Prediction
    arrow(10, 5, 8.5, 4)      # From GradCAM
    
    # === LEVEL 5: RAG ===
    box(4.5, 2.5, 5, 1.5, c_rag, 'Semantic RAG\nKnowledge Base + PubMed', 11)
    
    # Arrow down
    arrow(7, 2.5, 7, 1.5)
    
    # === LEVEL 6: OUTPUT ===
    box(4.5, 0, 5, 1.5, c_output, 'Explainable Output\nDiagnosis + Visual + Context', 10)
    
    # Legend
    legend_items = [
        mpatches.Patch(color=c_input, label='Input'),
        mpatches.Patch(color=c_preprocess, label='Preprocessing'),
        mpatches.Patch(color=c_model, label='CNN Model'),
        mpatches.Patch(color=c_xai, label='Explainability'),
        mpatches.Patch(color=c_rag, label='RAG'),
        mpatches.Patch(color=c_output, label='Output'),
    ]
    ax.legend(handles=legend_items, loc='upper right', fontsize=10, title='Components')
    
    plt.tight_layout()
    return fig


def create_simplified_diagram():
    """Create horizontal linear flow diagram."""
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Title
    ax.text(8, 3.6, 'LungXAI: CT Scan → Explainable Diagnosis', 
            fontsize=14, fontweight='bold', ha='center')
    
    # Pipeline stages
    stages = [
        ('CT Scan\n1,535 images', '#3498db'),
        ('Preprocess\n224×224', '#9b59b6'),
        ('MobileNetV2\n2.2M params', '#27ae60'),
        ('GradCAM\nHeatmap', '#e74c3c'),
        ('RAG\nContext', '#f39c12'),
        ('Output\nExplanation', '#1abc9c'),
    ]
    
    bw, bh = 2.2, 2.0  # box width, height
    gap = 0.4
    start_x = 0.5
    y = 0.8
    
    for i, (label, color) in enumerate(stages):
        x = start_x + i * (bw + gap)
        patch = FancyBboxPatch((x, y), bw, bh,
                               boxstyle="round,rounding_size=0.2",
                               facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(patch)
        ax.text(x + bw/2, y + bh/2, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        
        # Arrow to next box
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + bw + gap, y + bh/2), xytext=(x + bw, y + bh/2),
                        arrowprops=dict(arrowstyle='-|>', color='#2c3e50', lw=2))
    
    plt.tight_layout()
    return fig


def create_dataset_distribution():
    """Create bar chart of dataset class distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Actual data from archive(1)/Lung Cancer Dataset
    classes = ['Normal', 'Adenocarcinoma', 'Squamous\nCell', 'Large Cell', 'Benign']
    counts = [631, 337, 260, 187, 120]
    colors = ['#27ae60', '#3498db', '#9b59b6', '#e74c3c', '#f39c12']
    total = sum(counts)
    
    bars = ax.bar(classes, counts, color=colors, edgecolor='black', lw=1.5)
    
    # Add count and percentage on each bar
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        # Count on top
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count}', ha='center', fontsize=11, fontweight='bold')
        # Percentage inside
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                f'{pct:.1f}%', ha='center', va='center', fontsize=10, 
                color='white', fontweight='bold')
    
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_xlabel('Cancer Type', fontsize=12)
    ax.set_title(f'LungXAI Dataset: {total} CT Scan Images', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 700)
    
    plt.tight_layout()
    return fig


def create_data_split_chart():
    """Create train/validation/test split visualization."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    classes = ['Normal', 'Adenocarcinoma', 'Squamous Cell', 'Large Cell', 'Benign']
    train = [442, 236, 182, 131, 84]
    val = [95, 51, 39, 28, 18]
    test = [94, 50, 39, 28, 18]
    
    x = np.arange(len(classes))
    w = 0.25
    
    b1 = ax.bar(x - w, train, w, label=f'Train (1,075)', color='#27ae60', edgecolor='black')
    b2 = ax.bar(x, val, w, label=f'Val (231)', color='#3498db', edgecolor='black')
    b3 = ax.bar(x + w, test, w, label=f'Test (229)', color='#e74c3c', edgecolor='black')
    
    # Add labels
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 3, f'{int(h)}',
                    ha='center', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.set_title('Dataset Split: 70% Train / 15% Val / 15% Test (Total: 1,535)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 500)
    
    plt.tight_layout()
    return fig


def create_gradcam_flow():
    """Create GradCAM explanation flow diagram."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    ax.text(7, 5.6, 'GradCAM Visualization Flow', fontsize=14, fontweight='bold', ha='center')
    ax.text(7, 5.2, 'Gradient-weighted Class Activation Mapping', fontsize=11, ha='center', color='#666')
    
    # Flow stages
    stages = [
        ('CT Scan\nImage', '#3498db', 2.0),
        ('Forward\nPass', '#9b59b6', 2.0),
        ('Gradients\n(Backprop)', '#27ae60', 2.0),
        ('Weighted\nActivations', '#e74c3c', 2.0),
        ('Heatmap\nOverlay', '#f39c12', 2.0),
    ]
    
    x = 0.5
    y = 2.5
    h = 2.0
    gap = 0.4
    
    for i, (label, color, w) in enumerate(stages):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,rounding_size=0.2",
                               facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + w + gap, y + h/2), xytext=(x + w, y + h/2),
                        arrowprops=dict(arrowstyle='-|>', color='#2c3e50', lw=2))
        x += w + gap
    
    # Description
    ax.text(7, 1.0, 'Highlights regions that influenced model prediction',
            ha='center', fontsize=11, style='italic', color='#666')
    
    plt.tight_layout()
    return fig


def create_architecture_diagram():
    """Create MobileNetV2 architecture flow."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis('off')
    
    ax.text(7, 4.6, 'MobileNetV2 Architecture', fontsize=14, fontweight='bold', ha='center')
    ax.text(7, 4.2, 'Primary Model for LungXAI', fontsize=11, ha='center', color='#666')
    
    blocks = [
        ('Input\n224×224×3', '#3498db', 2.0),
        ('Conv2d\n32 ch', '#9b59b6', 1.5),
        ('Inverted\nResiduals', '#27ae60', 2.5),
        ('Conv2d\n1280 ch', '#27ae60', 1.5),
        ('AvgPool\n+FC', '#e74c3c', 1.5),
        ('5 Classes', '#1abc9c', 1.8),
    ]
    
    x = 0.3
    y = 1.5
    h = 2.0
    gap = 0.3
    
    for i, (label, color, w) in enumerate(blocks):
        patch = FancyBboxPatch((x, y), w, h,
                               boxstyle="round,rounding_size=0.15",
                               facecolor=color, edgecolor='black', lw=2)
        ax.add_patch(patch)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x + w + gap, y + h/2), xytext=(x + w, y + h/2),
                        arrowprops=dict(arrowstyle='-|>', color='#2c3e50', lw=2))
        x += w + gap
    
    # Summary
    ax.text(7, 0.7, 'Parameters: 2.2M | Baseline: 89.61% | Fine-tuned: 97.40%',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#d5f5e3', edgecolor='#27ae60'))
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images')
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating LungXAI diagrams...")
    
    diagrams = [
        ('pipeline_diagram.png', create_pipeline_diagram),
        ('pipeline_simplified.png', create_simplified_diagram),
        ('dataset_dist.png', create_dataset_distribution),
        ('data_split.png', create_data_split_chart),
        ('architecture.png', create_architecture_diagram),
        ('gradcam_flow.png', create_gradcam_flow),
    ]
    
    for name, func in diagrams:
        fig = func()
        path = os.path.join(output_dir, name)
        fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {name}")
    
    print(f"\nAll diagrams saved to: {output_dir}")
