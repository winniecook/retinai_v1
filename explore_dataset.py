#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def analyze_image(image_path):
    """Analyze properties of a single image"""
    img = Image.open(image_path)
    img_array = np.array(img)
    
    return {
        'mode': img.mode,
        'format': img.format,
        'size': img.size,
        'intensity': np.mean(img_array),
        'file_size': os.path.getsize(image_path) / (1024 * 1024)  # Size in MB
    }

def analyze_class_folder(folder_path):
    """Analyze all images in a class folder"""
    folder_stats = {
        'dimensions': [],
        'intensities': [],
        'file_sizes': [],
        'sample_props': {}
    }
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        img_path = os.path.join(folder_path, img_file)
        try:
            props = analyze_image(img_path)
            
            folder_stats['dimensions'].append(props['size'])
            folder_stats['intensities'].append(props['intensity'])
            folder_stats['file_sizes'].append(props['file_size'])
            
            # Store sample properties for the first image
            if not folder_stats['sample_props']:
                folder_stats['sample_props'] = props
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    return {
        'num_images': len(image_files),
        'unique_dimensions': set(folder_stats['dimensions']),
        'avg_intensity': np.mean(folder_stats['intensities']),
        'avg_file_size': np.mean(folder_stats['file_sizes']),
        'sample_props': folder_stats['sample_props']
    }

def create_visualizations(dataset_stats, output_dir):
    """Create and save dataset visualizations"""
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # 1. Class distribution pie chart
    plt.subplot(1, 3, 1)
    sizes = [stats['num_images'] for stats in dataset_stats.values()]
    labels = list(dataset_stats.keys())
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Class Distribution')
    
    # 2. Average intensities bar plot
    plt.subplot(1, 3, 2)
    intensities = [stats['avg_intensity'] for stats in dataset_stats.values()]
    plt.bar(labels, intensities)
    plt.title('Average Image Intensity by Class')
    plt.ylabel('Intensity')
    plt.xticks(rotation=45)
    
    # 3. File sizes bar plot
    plt.subplot(1, 3, 3)
    file_sizes = [stats['avg_file_size'] for stats in dataset_stats.values()]
    plt.bar(labels, file_sizes)
    plt.title('Average File Size by Class')
    plt.ylabel('Size (MB)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Set paths
    data_dir = os.path.expanduser("~/retinal_project2/data")
    class_folders = ['normal', 'cataract', 'glaucoma']
    
    print("Starting dataset analysis...\n")
    
    dataset_stats = {}
    total_images = 0
    
    # Analyze each class folder
    for folder in class_folders:
        print(f"\nAnalyzing {folder}...")
        folder_path = os.path.join(data_dir, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue
            
        stats = analyze_class_folder(folder_path)
        dataset_stats[folder] = stats
        total_images += stats['num_images']
        
        # Print sample image properties
        props = stats['sample_props']
        print("Sample image properties:")
        print(f"Mode: {props['mode']}")
        print(f"Format: {props['format']}")
        print(f"Size: {props['size']}")
    
    # Print dataset summary
    print("\nDataset Summary:")
    print("-" * 50)
    
    for folder, stats in dataset_stats.items():
        print(f"\n{folder}:")
        print(f"Number of images: {stats['num_images']}")
        print(f"Image dimensions: {stats['unique_dimensions']}")
        print(f"Average intensity: {stats['avg_intensity']:.2f}")
        print(f"Average file size: {stats['avg_file_size']:.2f} MB")
    
    print("\nClass Distribution:\n")
    for folder, stats in dataset_stats.items():
        percentage = (stats['num_images'] / total_images) * 100
        print(f"{folder.capitalize()}: {stats['num_images']} images ({percentage:.0f}%)")
    
    # Create and save visualizations
    create_visualizations(dataset_stats, data_dir)
    print(f"\nVisualizations saved to {os.path.join(data_dir, 'class_distribution.png')}")

if __name__ == "__main__":
    main()
