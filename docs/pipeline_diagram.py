# =============================================================================
# PIPELINE DIAGRAM GENERATOR
# Creates a visual diagram of the Explainable AI Pipeline for Faculty
# =============================================================================
"""
Generates a professional pipeline diagram for faculty presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def create_pipeline_diagram(save_path="pipeline_diagram.png"):
    """Create a professional pipeline diagram."""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    colors = {
        'input': '#3498db',      # Blue
        'process': '#2ecc71',    # Green
        'model': '#e74c3c',      # Red
        'xai': '#9b59b6',        # Purple
        'rag': '#f39c12',        # Orange
        'output': '#1abc9c',     # Teal
        'arrow': '#34495e',      # Dark gray
    }
    
    # Title
    ax.text(8, 9.5, 'Explainable AI for Lung Cancer Classification', 
            fontsize=20, fontweight='bold', ha='center', va='center',
            color='#2c3e50')
    ax.text(8, 9.0, 'Using Deep Learning and RAG-Based Knowledge Retrieval', 
            fontsize=14, ha='center', va='center', color='#7f8c8d', style='italic')
    
    # =========================================================================
    # PIPELINE BOXES
    # =========================================================================
    
    def draw_box(x, y, width, height, color, text, subtitle=""):
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.05,rounding_size=0.2",
                             facecolor=color, edgecolor='white', linewidth=3,
                             alpha=0.9)
        ax.add_patch(box)
        ax.text(x, y + 0.15, text, fontsize=12, fontweight='bold', 
                ha='center', va='center', color='white')
        if subtitle:
            ax.text(x, y - 0.25, subtitle, fontsize=9, 
                    ha='center', va='center', color='white', alpha=0.9)
    
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                                    lw=2.5, mutation_scale=20))
    
    # Row 1: Input
    draw_box(2, 7, 2.5, 1.2, colors['input'], 'CT Scan Image', '224×224 RGB')
    
    # Row 2: Preprocessing
    draw_box(5.5, 7, 2.5, 1.2, colors['process'], 'Preprocess', 'Resize, Normalize')
    
    # Row 3: Model
    draw_box(9, 7, 2.5, 1.2, colors['model'], 'ResNet-50', 'Transfer Learning')
    
    # Row 4: Prediction
    draw_box(12.5, 7, 2.5, 1.2, colors['output'], 'Prediction', '5-Class + Confidence')
    
    # Arrows Row 1
    draw_arrow(3.3, 7, 4.2, 7)
    draw_arrow(6.8, 7, 7.7, 7)
    draw_arrow(10.3, 7, 11.2, 7)
    
    # =========================================================================
    # XAI BRANCH
    # =========================================================================
    
    # Arrow down from model
    draw_arrow(9, 6.3, 9, 5.5)
    
    draw_box(9, 4.8, 2.5, 1.2, colors['xai'], 'Grad-CAM', 'Visual Explanation')
    
    # Arrow down
    draw_arrow(9, 4.1, 9, 3.3)
    
    draw_box(9, 2.6, 2.8, 1.2, colors['xai'], 'Heatmap Analysis', 'Region Detection')
    
    # =========================================================================
    # RAG BRANCH
    # =========================================================================
    
    # Arrow from heatmap to RAG
    draw_arrow(10.5, 2.6, 11.5, 2.6)
    
    draw_box(13, 2.6, 2.5, 1.2, colors['rag'], 'RAG Module', 'Knowledge Retrieval')
    
    # Knowledge Base
    draw_box(13, 4.8, 2.5, 1.0, colors['rag'], 'Knowledge Base', '19 Medical Entries')
    draw_arrow(13, 4.2, 13, 3.3)
    
    # =========================================================================
    # OUTPUT - positioned lower with more gap
    # =========================================================================
    
    # Arrow from prediction down
    draw_arrow(12.5, 6.3, 12.5, 5.5)
    
    # Final output box - positioned lower for more gap
    output_box = FancyBboxPatch((6, -0.3), 5.5, 1.6,
                                 boxstyle="round,pad=0.05,rounding_size=0.3",
                                 facecolor=colors['output'], edgecolor='white', 
                                 linewidth=3, alpha=0.9)
    ax.add_patch(output_box)
    ax.text(8.75, 0.8, 'EXPLAINABLE OUTPUT', fontsize=13, fontweight='bold', 
            ha='center', va='center', color='white')
    ax.text(8.75, 0.35, '• Prediction + Confidence   • Grad-CAM Heatmap   • Medical Explanation', fontsize=9, 
            ha='center', va='center', color='white')
    
    # Arrow from Heatmap Analysis - straight down to output
    draw_arrow(9, 1.9, 9, 1.35)
    
    # Arrow from RAG module - straight down to output  
    draw_arrow(13, 1.9, 11.5, 0.8)
    
    # Arrow from Prediction - goes right, then down along right edge, then to output
    # This creates an L-shaped path: right -> down -> left to output
    # Draw as line segments
    ax.plot([12.5, 15.2], [5.5, 5.5], color=colors['arrow'], lw=2.5)  # Right
    ax.plot([15.2, 15.2], [5.5, 0.5], color=colors['arrow'], lw=2.5)  # Down
    ax.annotate('', xy=(11.5, 0.5), xytext=(15.2, 0.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], 
                                lw=2.5, mutation_scale=20))  # Left to output with arrow
    
    # =========================================================================
    # LEGEND
    # =========================================================================
    
    legend_y = 2.5
    legend_x = 1.5
    
    ax.text(legend_x, 4.5, 'Components:', fontsize=11, fontweight='bold', color='#2c3e50')
    
    components = [
        (colors['input'], 'Input Layer'),
        (colors['process'], 'Preprocessing'),
        (colors['model'], 'Deep Learning Model'),
        (colors['xai'], 'Explainable AI (XAI)'),
        (colors['rag'], 'RAG Knowledge Retrieval'),
        (colors['output'], 'Output'),
    ]
    
    for i, (color, label) in enumerate(components):
        rect = plt.Rectangle((legend_x - 0.3, 4.0 - i*0.45), 0.4, 0.3, 
                              facecolor=color, edgecolor='white', linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + 0.4, 4.15 - i*0.45, label, fontsize=10, 
                va='center', color='#2c3e50')
    
    # =========================================================================
    # PHASE LABELS
    # =========================================================================
    
    # Phase annotations
    ax.text(2, 8.2, 'PHASE 1', fontsize=9, ha='center', color='#7f8c8d', fontweight='bold')
    ax.text(5.5, 8.2, 'PHASE 2', fontsize=9, ha='center', color='#7f8c8d', fontweight='bold')
    ax.text(9, 8.2, 'PHASE 3', fontsize=9, ha='center', color='#7f8c8d', fontweight='bold')
    ax.text(9, 5.5, 'PHASE 4', fontsize=9, ha='center', color='#7f8c8d', fontweight='bold')
    ax.text(13, 5.5, 'PHASE 5', fontsize=9, ha='center', color='#7f8c8d', fontweight='bold')
    ax.text(8.75, -0.7, 'PHASE 6', fontsize=9, ha='center', color='#7f8c8d', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✓ Pipeline diagram saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, r'd:\Major Project')
    
    # Create docs directory
    os.makedirs(r'd:\Major Project\docs', exist_ok=True)
    
    # Generate diagram
    fig = create_pipeline_diagram(r'd:\Major Project\docs\pipeline_diagram.png')
    plt.show()
