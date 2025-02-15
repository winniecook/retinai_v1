"""
Retinal Disease Dataset Handler

This module implements the dataset handling functionality for the retinal disease classification project.
It includes data loading, preprocessing, augmentation, and splitting into train/val/test sets.

Author: Winnie Cook
Date: February 2025
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from pathlib import Path

class RetinalDataset(Dataset):
    """
    Custom Dataset class for retinal disease images.
    
    Handles loading and preprocessing of retinal images for classification.
    Supports train/val/test splits and implements data augmentation for training.
    
    Args:
        data_dir (str): Root directory containing class-specific subdirectories
        mode (str): One of 'train', 'val', 'test', or 'all'
        img_size (int): Target size for image resizing
    """
    def __init__(self, data_dir, mode='train', img_size=224):
        self.data_dir = Path(data_dir)
        self.mode = mode
        self.img_size = img_size
        self.images, self.labels = self._get_data()
        self.transform = self._get_transforms()
        if mode == 'train':
            self._print_stats()

    def _get_data(self):
        """
        Load image paths and labels from the directory structure.
        
        Expects a directory structure where each class has its own subdirectory:
        data_dir/
            normal/
                image1.png
                image2.png
            cataract/
                image1.png
                ...
            glaucoma/
                image1.png
                ...
                
        Returns:
            tuple: (image_paths, labels) as numpy arrays
        """
        images = []
        labels = []
        class_map = {'normal': 0, 'cataract': 1, 'glaucoma': 2}
        
        # Load images from each class directory
        for class_name in class_map.keys():
            class_dir = self.data_dir / class_name
            class_images = [str(class_dir / img)
                          for img in os.listdir(class_dir)
                          if img.endswith('.png')]
            images.extend(class_images)
            labels.extend([class_map[class_name]] * len(class_images))

        images = np.array(images)
        labels = np.array(labels)

        # Split data if not using all data
        if self.mode != 'all':
            # First split: 80% train, 20% temp (for val and test)
            train_images, temp_images, train_labels, temp_labels = train_test_split(
                images, labels, test_size=0.2, stratify=labels, random_state=42
            )
            
            if self.mode == 'train':
                return train_images, train_labels
            
            # Second split: Split temp into val and test (50% each, so 10% of total each)
            val_images, test_images, val_labels, test_labels = train_test_split(
                temp_images, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
            )
            
            if self.mode == 'val':
                return val_images, val_labels
            elif self.mode == 'test':
                return test_images, test_labels

        return images, labels

    def _get_transforms(self):
        """
        Get image transformations based on the dataset mode.
        
        Training mode includes data augmentation transforms.
        Validation and test modes only include necessary preprocessing.
        
        Returns:
            transforms.Compose: Composition of image transformations
        """
        if self.mode == 'train':
            return transforms.Compose([
                transforms.RandomResizedCrop(self.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def _print_stats(self):
        """Print dataset statistics including class distribution."""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        print("\nDataset Statistics:")
        print(f"Total images: {len(self.labels)}")
        for label, count in zip(unique_labels, counts):
            print(f"Class {label}: {count} images")

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch
            
        Returns:
            dict: Contains 'image' (tensor) and 'label' (int)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return {'image': image, 'label': label}

def get_data_loaders(data_dir, batch_size=16, num_workers=4):
    """
    Create DataLoader instances for train, validation, and test sets.
    
    Args:
        data_dir (str): Root directory containing the dataset
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, class_weights)
    """
    train_dataset = RetinalDataset(data_dir, mode='train')
    val_dataset = RetinalDataset(data_dir, mode='val')
    test_dataset = RetinalDataset(data_dir, mode='test')
    
    # Calculate class weights for handling class imbalance
    labels = np.array(train_dataset.labels)
    class_counts = np.bincount(labels)
    total = len(labels)
    class_weights = torch.FloatTensor(total / (len(class_counts) * class_counts))
    
    sample_weights = class_weights[labels]
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights