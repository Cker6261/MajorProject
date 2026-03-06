"""
Generate clean, IEEE-style hyperparameter table images for research paper.
Format: Model | Hyperparameters | Range | Result
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Purple color scheme matching the example
HEADER_COLOR = '#8B5CA3'  # Purple header
ROW_LIGHT = '#F5F0F8'     # Light purple/lavender
ROW_WHITE = '#FFFFFF'     # White


def create_hyperparameter_table():
    """Create a clean IEEE-style table: Model | Hyperparameters | Range | Result"""
    
    # Data in requested format
    data = [
        ['★ MobileNetV2', 'LR, WD, Dropout, Optimizer', '1e-4, 1e-4, 0.5, AdamW', 'Acc: 97.40%, F1: 97.40%'],
        ['ResNet-50', 'LR, WD, Dropout, Optimizer', '1e-4, 1e-4, 0.5, AdamW', 'Acc: 96.97%, F1: 96.95%'],
        ['DenseNet-121', 'LR, WD, Dropout, Optimizer', '1e-3, 0, 0.5, Adam', 'Acc: 84.42%, F1: 82.63%'],
        ['EfficientNet-B0', 'LR, WD, Dropout, Optimizer', '1e-3, 0, 0.5, Adam', 'Acc: 72.29%, F1: 72.62%'],
        ['VGG-16', 'LR, WD, Dropout, Optimizer', '1e-3, 0, 0.5, Adam', 'Acc: 71.43%, F1: 69.39%'],
    ]
    
    headers = ['Model', 'Hyperparameters', 'Range', 'Result']
    
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.axis('off')
    
    # Title
    ax.set_title('Table VI: CNN Model Hyperparameters', fontsize=12, fontweight='bold', 
                 pad=10, loc='center')
    
    # Create table
    table = ax.table(
        cellText=[headers] + data,
        colWidths=[0.22, 0.28, 0.26, 0.24],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header row - purple background, white text
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor('white')
        cell.set_linewidth(1)
    
    # Style data rows - alternating white and light purple
    for i in range(5):
        row_color = ROW_LIGHT if i % 2 == 0 else ROW_WHITE
        for j in range(4):
            cell = table[(i + 1, j)]
            cell.set_facecolor(row_color)
            cell.set_text_props(fontsize=10, color='black')
            cell.set_edgecolor('#DDDDDD')
            cell.set_linewidth(0.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'images', 'hyperparameter_table.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Table image saved to: {output_path}")
    return output_path


def create_summary_table():
    """Create a clean IEEE-style summary table with performance metrics."""
    
    # Data: Model | Accuracy | Precision | Recall | F1-Score
    data = [
        ['★ MobileNetV2', '97.40%', '97.50%', '97.40%', '97.40%'],
        ['ResNet-50', '96.97%', '96.99%', '96.97%', '96.95%'],
        ['DenseNet-121', '84.42%', '85.70%', '84.42%', '82.63%'],
        ['EfficientNet-B0', '72.29%', '73.58%', '72.29%', '72.62%'],
        ['VGG-16', '71.43%', '69.82%', '71.43%', '69.39%'],
    ]
    
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axis('off')
    
    # Title
    ax.set_title('Table VII: CNN Model Performance Comparison', fontsize=12, fontweight='bold', 
                 pad=10, loc='center')
    
    table = ax.table(
        cellText=[headers] + data,
        colWidths=[0.28, 0.16, 0.16, 0.16, 0.16],
        loc='center',
        cellLoc='center'
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    
    # Style header row
    for j in range(5):
        cell = table[(0, j)]
        cell.set_facecolor(HEADER_COLOR)
        cell.set_text_props(color='white', fontweight='bold', fontsize=10)
        cell.set_edgecolor('white')
        cell.set_linewidth(1)
    
    # Style data rows
    for i in range(5):
        row_color = ROW_LIGHT if i % 2 == 0 else ROW_WHITE
        for j in range(5):
            cell = table[(i + 1, j)]
            cell.set_facecolor(row_color)
            cell.set_text_props(fontsize=10, color='black')
            cell.set_edgecolor('#DDDDDD')
            cell.set_linewidth(0.5)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), 'images', 'model_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Summary table saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    create_hyperparameter_table()
    create_summary_table()
    print("\n✓ All table images generated successfully!")
