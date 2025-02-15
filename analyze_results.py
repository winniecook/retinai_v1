"""
Results Analysis Module

This module provides functionality for analyzing and visualizing training results,
including learning curves, confusion matrices, and classification metrics.

Author: Winnie Cook
Date: February 2025
"""

import torch
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
import pandas as pd

def load_training_log(output_dir):
    """
    Load and parse training metrics from the training log file.
    
    Args:
        output_dir (str): Directory containing the training.log file
        
    Returns:
        dict: Dictionary containing lists of training and validation metrics
    """
    log_file = Path(output_dir) / 'training.log'
    metrics = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    with open(log_file) as f:
        for line in f:
            if 'Epoch' in line and 'Train Loss' in line:
                parts = line.split()
                metrics['train_loss'].append(float(parts[parts.index('Loss:')+1].rstrip(',')))
                metrics['train_acc'].append(float(parts[parts.index('Acc:')+1].rstrip(',')))
                metrics['val_loss'].append(float(parts[parts.index('Loss:', parts.index('Loss:')+1)+1].rstrip(',')))
                metrics['val_acc'].append(float(parts[parts.index('Acc:', parts.index('Acc:')+1)+1]))
    
    return metrics

def plot_training_curves(metrics, output_dir):
    """
    Plot training and validation loss/accuracy curves.
    
    Creates a figure with two subplots showing the progression of
    loss and accuracy metrics throughout training.
    
    Args:
        metrics (dict): Dictionary containing training metrics
        output_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(12, 5))
    
    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_loss'], label='Train')
    plt.plot(metrics['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(metrics['train_acc'], label='Train')
    plt.plot(metrics['val_acc'], label='Validation')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_curves.png')

def analyze_confusion_matrix(output_dir):
    """
    Display and analyze the confusion matrix.
    
    Loads the pre-generated confusion matrix image and displays it
    with additional analysis of class-wise performance.
    
    Args:
        output_dir (str): Directory containing the confusion matrix image
    """
    cm_path = Path(output_dir) / 'metrics' / 'confusion_matrix.png'
    plt.figure(figsize=(10, 8))
    img = plt.imread(cm_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title('Confusion Matrix Analysis')
    plt.savefig(Path(output_dir) / 'confusion_matrix_analysis.png')

def main():
    """
    Main function to run the analysis pipeline.
    
    Performs the following analyses:
    1. Loads and plots training metrics
    2. Analyzes confusion matrix
    3. Generates comprehensive performance report
    """
    output_dir = '../outputs/20250124_113559'  # Update with your timestamp
    
    # Load and plot training metrics
    metrics = load_training_log(output_dir)
    plot_training_curves(metrics, output_dir)
    
    # Analyze confusion matrix
    analyze_confusion_matrix(output_dir)
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"Best Validation Accuracy: {max(metrics['val_acc']):.4f}")
    print(f"Final Training Loss: {metrics['train_loss'][-1]:.4f}")
    print(f"Final Validation Loss: {metrics['val_loss'][-1]:.4f}")

if __name__ == '__main__':
    main()