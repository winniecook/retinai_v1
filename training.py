"""
Training Pipeline for Retinal Disease Classification

This module implements a comprehensive training pipeline for the RetinalClassifier model,
including training loop management, metrics tracking, checkpointing, and evaluation.

Author: Winnie Cook
Date: February 2025
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from typing import Dict, List
import logging
from pathlib import Path
import wandb
from tqdm import tqdm

class TrainingPipeline:
    """
    Comprehensive training pipeline for the RetinalClassifier model.
    
    Handles the entire training process including:
    - Training and validation loops
    - Metrics tracking and logging
    - Model checkpointing
    - Early stopping
    - Results visualization
    
    Args:
        model: The RetinalClassifier model
        trainer: ModelTrainer instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        test_loader: DataLoader for test data
        config: Dictionary containing training configuration
        output_dir: Directory for saving outputs
    """
    def __init__(
        self,
        model: torch.nn.Module,
        trainer: 'ModelTrainer',
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: Dict,
        output_dir: str = 'outputs'
    ):
        self.model = model
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.output_dir = Path(output_dir)
        
        # Create necessary directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'models').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        
        # Initialize logging and metrics tracking
        self.setup_logging()
        self.train_losses = []
        self.val_losses = []
        self.best_epoch = 0
    
    def setup_logging(self):
        """Configure logging to both file and console output."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train(self):
        """
        Main training loop.
        
        Executes the training process for the specified number of epochs,
        handling model checkpointing, early stopping, and metrics logging.
        """
        self.logger.info("Starting training...")
        for epoch in range(self.config['epochs']):
            self.trainer.current_epoch = epoch
            
            # Training and validation phases
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)
            
            # Track losses
            self.train_losses.append(train_metrics['loss'])
            self.val_losses.append(val_metrics['loss'])
            
            # Learning rate scheduling
            if self.trainer.scheduler is not None:
                self.trainer.scheduler.step(val_metrics['loss'])
            
            # Model checkpointing
            if val_metrics['loss'] < self.trainer.best_val_loss:
                self.best_epoch = epoch
                self.trainer.best_val_loss = val_metrics['loss']
                self.trainer.best_model_state = self.model.state_dict()
                self.save_checkpoint(epoch, val_metrics['loss'], 'best_model.pth')
            
            # Early stopping check
            if epoch - self.best_epoch > self.config['patience']:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # Periodic checkpointing
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics['loss'], f'checkpoint_epoch_{epoch+1}.pth')
            
            # Log metrics
            self.log_metrics(epoch, train_metrics, val_metrics)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one epoch of training.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            dict: Dictionary containing training metrics
        """
        self.model.train()
        epoch_metrics = {'loss': 0.0, 'acc': 0.0}
        
        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self.trainer.train_step(batch)
            for k, v in metrics.items():
                epoch_metrics[k] += v
            
        # Calculate average metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(self.train_loader)
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Execute one epoch of validation.
        
        Args:
            epoch (int): Current epoch number
            
        Returns:
            dict: Dictionary containing validation metrics
        """
        self.model.eval()
        val_metrics = {'loss': 0.0, 'acc': 0.0}
        
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = self.trainer.validate_step(batch)
                for k, v in metrics.items():
                    val_metrics[k] += v
        
        # Calculate average metrics
        for k in val_metrics:
            val_metrics[k] /= len(self.val_loader)
        return val_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the test dataset.
        
        Returns:
            dict: Dictionary containing test metrics
        """
        self.logger.info("Starting evaluation...")
        self.model.eval()
        all_preds = []
        all_targets = []
        test_loss = 0
        
        with torch.no_grad():
            for batch in self.test_loader:
                images = batch['image'].to(self.trainer.device)
                targets = batch['label'].to(self.trainer.device)
                outputs, _ = self.model(images)
                loss = self.trainer.criterion(outputs, targets)
                test_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        test_loss /= len(self.test_loader)
        self.plot_metrics(all_targets, all_preds)
        report = classification_report(
            all_targets, 
            all_preds,
            target_names=['Normal', 'Cataract', 'Glaucoma'],
            output_dict=True
        )
        
        return {'test_loss': test_loss, 'report': report}
    
    def plot_metrics(self, targets, preds):
        """
        Plot confusion matrix and ROC curves.
        
        Args:
            targets (list): List of true labels
            preds (list): List of predicted labels
        """
        # Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(targets, preds)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Cataract', 'Glaucoma'],
            yticklabels=['Normal', 'Cataract', 'Glaucoma']
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.output_dir / 'metrics' / 'confusion_matrix.png')
        plt.close()
        
        # ROC Curves
        plt.figure(figsize=(10, 8))
        targets_onehot = np.eye(3)[targets]
        preds_onehot = np.eye(3)[preds]
        
        for i, (label, color) in enumerate(zip(
            ['Normal', 'Cataract', 'Glaucoma'], 
            ['blue', 'red', 'green']
        )):
            fpr, tpr, _ = roc_curve(targets_onehot[:, i], preds_onehot[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=color, label=f'{label} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.savefig(self.output_dir / 'metrics' / 'roc_curves.png')
        plt.close()
    
    def save_checkpoint(self, epoch, val_loss, filename):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch number
            val_loss (float): Validation loss
            filename (str): Checkpoint filename
        """
        checkpoint_path = self.output_dir / 'models' / filename
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'val_loss': val_loss,
        }, checkpoint_path)
    
    def log_metrics(self, epoch, train_metrics, val_metrics):
        """
        Log training and validation metrics.
        
        Args:
            epoch (int): Current epoch number
            train_metrics (dict): Training metrics
            val_metrics (dict): Validation metrics
        """
        msg = (f"Epoch {epoch}: "
               f"Train Loss: {train_metrics['loss']:.4f}, "
               f"Train Acc: {train_metrics['acc']:.4f}, "
               f"Val Loss: {val_metrics['loss']:.4f}, "
               f"Val Acc: {val_metrics['acc']:.4f}")
        self.logger.info(msg)