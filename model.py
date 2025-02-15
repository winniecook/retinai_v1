"""
DenseNet-based Retinal Disease Classification Model

This module implements a retinal disease classifier using DenseNet121 as the backbone.
The implementation leverages transfer learning and includes a custom classifier head
for improved performance on retinal image classification tasks.

Author: Winnie Cook
Date: February 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple

class RetinalClassifier(nn.Module):
    """
    A neural network model for retinal disease classification based on DenseNet121.
    
    The model uses transfer learning from ImageNet pre-trained weights and adds a
    custom classifier head with dropout for regularization. The architecture is
    specifically tuned for retinal image classification tasks.
    
    Args:
        num_classes (int): Number of disease categories to classify
        pretrained (bool): Whether to use ImageNet pre-trained weights
    """
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        # Initialize the base DenseNet121 model
        self.base_model = models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        in_features = self.base_model.classifier.in_features  # 1024 features
        
        # Custom classifier head with increased capacity and dropout
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            tuple: (predictions, None) where predictions are class logits
        """
        return self.base_model(x), None

class ModelTrainer:
    """
    Training wrapper for the RetinalClassifier model.
    
    Handles training loop, optimization, and model state management.
    Supports learning rate scheduling and maintains best model state.
    
    Args:
        model (nn.Module): The RetinalClassifier model
        criterion: Loss function
        optimizer: Optimization algorithm
        scheduler: Learning rate scheduler (optional)
        device (str): Device to run the model on ('cpu' or 'cuda')
    """
    def __init__(self, model, criterion, optimizer, scheduler=None, device='cpu'):
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None
    
    def train_step(self, batch):
        """
        Performs a single training step.
        
        Args:
            batch (dict): Dictionary containing 'image' and 'label' tensors
            
        Returns:
            dict: Dictionary containing training metrics (loss and accuracy)
        """
        self.model.train()
        data, targets = batch['image'].to(self.device), batch['label'].to(self.device)
        outputs, _ = self.model(data)
        loss = self.criterion(outputs, targets)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Calculate accuracy
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).float().mean()
        return {'loss': loss.item(), 'acc': acc.item()}
    
    @torch.no_grad()
    def validate_step(self, batch):
        """
        Performs a single validation step.
        
        Args:
            batch (dict): Dictionary containing 'image' and 'label' tensors
            
        Returns:
            dict: Dictionary containing validation metrics (loss and accuracy)
        """
        self.model.eval()
        data, targets = batch['image'].to(self.device), batch['label'].to(self.device)
        outputs, _ = self.model(data)
        loss = self.criterion(outputs, targets)
        preds = torch.argmax(outputs, dim=1)
        acc = (preds == targets).float().mean()
        return {'loss': loss.item(), 'acc': acc.item()}

def create_model(num_classes=3, learning_rate=1e-4, device='cpu'):
    """
    Factory function to create a new model instance with optimizer and scheduler.
    
    Args:
        num_classes (int): Number of disease categories
        learning_rate (float): Initial learning rate for the optimizer
        device (str): Device to run the model on
        
    Returns:
        tuple: (model, trainer) initialized and ready for training
    """
    model = RetinalClassifier(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    trainer = ModelTrainer(model, criterion, optimizer, scheduler, device)
    return model, trainer
