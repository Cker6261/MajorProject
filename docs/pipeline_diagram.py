"""
LungXAI Pipeline Diagram Generator
Generates a visual diagram of the complete explainable AI pipeline.

Models: MobileNetV2 (primary) with 4 CNN baselines (ResNet-50, VGG-16, DenseNet-121, EfficientNet-B0)
Uses ACTUAL trained/tested baseline results.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_diagram():
    """Create the main pipeline diagram with all components."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Color scheme
    colors = {
        'input': '#3498db',       # Blue
        'preprocess': '#9b59b6',  # Purple
        'model': '#2ecc71',       # Green (MobileNetV2)
        'xai': '#e74c3c',         # Red
        'rag': '#f39c12',         # Orange
        'output': '#1abc9c',      # Teal
        'knowledge': '#e67e22',   # Dark Orange
        'arrow': '#2c3e50',       # Dark gray
    }
    
    # Helper function to draw boxes
    def draw_box(x, y, width, height, color, text, fontsize=11, alpha=0.9):
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color, edgecolor='black', linewidth=2, alpha=alpha
        )
        ax.add_patch(box)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white',
                wrap=True)
    
    # Helper function to draw arrows
    def draw_arrow(start, end, color='#2c3e50'):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color=color, lw=2.5))
    
    # Title
    ax.text(8, 11.5, 'LungXAI: Explainable AI Pipeline for Lung Cancer Classification',
            ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(8, 11.0, 'CNN-Based Architecture with GradCAM & Semantic RAG',
            ha='center', va='center', fontsize=12, fontstyle='italic', color='#555')
    
    # ========== INPUT SECTION ==========
    draw_box(0.5, 8.5, 2.5, 1.5, colors['input'], 'CT Scan\nImage\n(Input)', fontsize=12)
    
    # ========== PREPROCESSING ==========
    draw_box(4, 8.5, 2.5, 1.5, colors['preprocess'], 'Preprocessing\n224×224\nNormalize', fontsize=11)
    draw_arrow((3.0, 9.25), (4.0, 9.25))
    
    # ========== MODEL SECTION ==========
    # Main model box
    draw_box(7.5, 8.0, 3.5, 2.5, colors['model'], 'MobileNetV2\n(Primary CNN)\n\n2.2M Parameters\nFine-tuned', fontsize=11)
    draw_arrow((6.5, 9.25), (7.5, 9.25))
    
    # CNN Baselines box (smaller, to the side)
    draw_box(11.5, 8.5, 2.5, 1.5, '#95a5a6', 'CNN Baselines\nResNet-50\nVGG-16\nDenseNet-121\nEfficientNet-B0', fontsize=8)
    ax.annotate('', xy=(11.5, 9.25), xytext=(11.0, 9.25),
                arrowprops=dict(arrowstyle='<->', color='#7f8c8d', lw=1.5, linestyle='dashed'))
    
    # ========== PREDICTION OUTPUT ==========
    draw_box(7.5, 5.5, 1.5, 1.5, colors['output'], 'Predict\nClass', fontsize=11)
    draw_arrow((9.25, 8.0), (8.25, 7.0))
    
    # ========== GRAD-CAM SECTION ==========
    draw_box(10, 5.5, 2.0, 1.5, colors['xai'], 'GradCAM\nHeatmap', fontsize=11)
    draw_arrow((9.5, 8.0), (10.5, 7.0))
    
    # Heatmap analysis
    draw_box(10, 3.5, 2.0, 1.5, colors['xai'], 'Analyze\nRegions', fontsize=10)
    draw_arrow((11, 5.5), (11, 5.0))
    
    # ========== SEMANTIC RAG SECTION ==========
    # Knowledge base
    draw_box(5, 3.5, 2.5, 1.5, colors['knowledge'], 'Knowledge\nBase\n(50+ Entries)', fontsize=10)
    
    # Semantic search
    draw_box(5, 1.5, 2.5, 1.5, colors['rag'], 'Semantic\nSearch\n(all-MiniLM-L6-v2)', fontsize=9)
    draw_arrow((6.25, 3.5), (6.25, 3.0))
    
    # PubMed
    draw_box(8.5, 1.5, 2.0, 1.5, colors['rag'], 'PubMed\nRetrieval', fontsize=10)
    
    # Connect prediction to semantic search
    draw_arrow((8.25, 5.5), (7.0, 4.25))
    ax.annotate('', xy=(6.25, 5.0), xytext=(8.0, 5.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2, 
                                connectionstyle='arc3,rad=-0.2'))
    
    # Connect heatmap analysis to semantic search
    draw_arrow((10, 4.25), (7.5, 3.0))
    
    # Connect semantic search to PubMed
    draw_arrow((7.5, 2.25), (8.5, 2.25))
    
    # ========== FINAL OUTPUT ==========
    draw_box(2, 1.5, 2.5, 1.5, colors['output'], 'Explainable\nOutput', fontsize=12)
    draw_arrow((5, 2.25), (4.5, 2.25))
    
    # Output details box
    output_text = """• Prediction + Confidence
• GradCAM Visualization
• Medical Context
• Evidence-Based Citations"""
    
    ax.text(0.5, 0.5, output_text, fontsize=9, va='top', ha='left',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', edgecolor='#bdc3c7'),
            family='monospace')
    
    # ========== LEGEND ==========
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input'),
        mpatches.Patch(color=colors['preprocess'], label='Preprocessing'),
        mpatches.Patch(color=colors['model'], label='CNN Model'),
        mpatches.Patch(color=colors['xai'], label='Explainability (XAI)'),
        mpatches.Patch(color=colors['rag'], label='Semantic RAG'),
        mpatches.Patch(color=colors['output'], label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              framealpha=0.9, title='Components')
    
    # ========== ANNOTATIONS ==========
    # Model specs (ACTUAL trained results)
    ax.text(14.5, 6.5, 'Baseline Results:', fontsize=10, fontweight='bold')
    specs = [
        'MobileNetV2: 89.61% (2.2M)',
        'DenseNet-121: 84.42% (7.0M)',
        'ResNet-50: 78.79% (23.5M)',
        'EfficientNet-B0: 72.29% (5.3M)',
        'VGG-16: 71.43% (138M)',
        '',
        'Fine-tuned:',
        'MobileNetV2: 97.40%',
        'ResNet-50: 96.97%',
    ]
    for i, spec in enumerate(specs):
        ax.text(14.5, 6.0 - i*0.35, spec, fontsize=8, family='monospace')
    
    plt.tight_layout()
    return fig

def create_simplified_diagram():
    """Create a simplified linear pipeline diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # Colors
    colors = ['#3498db', '#9b59b6', '#2ecc71', '#e74c3c', '#f39c12', '#1abc9c']
    labels = ['CT Scan', 'Preprocess', 'MobileNetV2', 'GradCAM', 'Semantic RAG', 'Explanation']
    details = ['Input', '224×224', '2.2M params', 'Heatmap', '50+ entries', 'Output']
    
    box_width = 2.0
    box_height = 2.0
    start_x = 0.5
    spacing = 2.3
    
    for i, (label, detail, color) in enumerate(zip(labels, details, colors)):
        x = start_x + i * spacing
        box = FancyBboxPatch(
            (x, 1), box_width, box_height,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color, edgecolor='black', linewidth=2, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x + box_width/2, 2.2, label, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax.text(x + box_width/2, 1.6, detail, ha='center', va='center',
                fontsize=9, color='white')
        
        if i < len(labels) - 1:
            ax.annotate('', xy=(x + spacing, 2), xytext=(x + box_width, 2),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
    
    ax.text(7, 3.7, 'LungXAI Pipeline: CT Scan → Explanation', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def create_model_comparison_diagram():
    """Create a diagram comparing CNN model architectures."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title - MobileNetV2 only (as requested)
    ax.text(6, 7.5, 'MobileNetV2 Architecture', 
            ha='center', fontsize=14, fontweight='bold')
    ax.text(6, 7.0, '(Primary Model for LungXAI)',
            ha='center', fontsize=11, fontstyle='italic', color='#555')
    
    # MobileNetV2 architecture blocks
    blocks = [
        ('Input\n224×224×3', '#3498db', 1.5),
        ('Conv2d\n32 filters', '#9b59b6', 1.2),
        ('Inverted\nResidual\nBlocks', '#2ecc71', 2.0),
        ('Conv2d\n1280', '#2ecc71', 1.2),
        ('AvgPool\n+ FC', '#e74c3c', 1.2),
        ('Output\n5 classes', '#1abc9c', 1.5),
    ]
    
    x_pos = 0.5
    y_pos = 4
    
    for i, (name, color, width) in enumerate(blocks):
        box = FancyBboxPatch(
            (x_pos, y_pos), width, 2.0,
            boxstyle="round,pad=0.02,rounding_size=0.15",
            facecolor=color, edgecolor='black', 
            linewidth=2, alpha=0.9
        )
        ax.add_patch(box)
        ax.text(x_pos + width/2, y_pos + 1.0, name, ha='center', va='center',
                fontsize=9, fontweight='bold', color='white')
        
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x_pos + width + 0.3, y_pos + 1), xytext=(x_pos + width, y_pos + 1),
                        arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
        x_pos += width + 0.4
    
    # Model summary
    ax.text(6, 2.5, 'Parameters: 2.2M | Accuracy: 97.40% | Inference: 5ms',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='#d5f5e3', edgecolor='#2ecc71'))
    ax.text(6, 1.8, 'Transfer Learning Gain: +7.79% (89.61% → 97.40%)',
            ha='center', fontsize=10, color='#27ae60')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    import os
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate main pipeline diagram
    fig = create_pipeline_diagram()
    fig.savefig(os.path.join(output_dir, 'pipeline_diagram.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: pipeline_diagram.png")
    
    # Generate simplified diagram
    fig2 = create_simplified_diagram()
    fig2.savefig(os.path.join(output_dir, 'pipeline_simplified.png'), 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: pipeline_simplified.png")
    
    # Generate model comparison
    fig3 = create_model_comparison_diagram()
    fig3.savefig(os.path.join(output_dir, 'model_comparison_diagram.png'), 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: model_comparison_diagram.png")
    
    plt.close('all')
    print("\nAll diagrams generated successfully!")
