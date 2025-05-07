import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import yaml
from datetime import datetime

from ..data.dataset import BrainScoreDataset
from ..models.model import BrainScoreModel
from .metrics import calculate_metrics

class Trainer:
    def __init__(
        self,
        config_path: str,
        device: str = "cuda"
    ):
        """
        Trainer class for the brain score prediction model.
        
        Args:
            config_path: Path to configuration file
            device: Device to run training on
        """
        self.device = device
        self.config = self._load_config(config_path)
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = nn.MSELoss()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def setup_model(self):
        """Initialize model and optimizer."""
        self.model = BrainScoreModel(
            num_demographic_features=len(self.config['model']['demographic_features']),
            num_cognitive_scores=len(self.config['model']['output_features']),
            dropout_rate=self.config['model']['dropout_rate']
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch in tqdm(train_loader, desc="Training"):
            # Move data to device
            image = batch['image'].to(self.device)
            demographic = batch['demographic'].to(self.device)
            cognitive_scores = batch['cognitive_scores'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(image, demographic)
            loss = self.criterion(outputs, cognitive_scores)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Record metrics
            total_loss += loss.item()
            predictions.append(outputs.detach().cpu().numpy())
            targets.append(cognitive_scores.detach().cpu().numpy())
            
        # Calculate metrics
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        metrics = calculate_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(train_loader)
        
        return metrics
    
    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on validation/test set.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move data to device
            image = batch['image'].to(self.device)
            demographic = batch['demographic'].to(self.device)
            cognitive_scores = batch['cognitive_scores'].to(self.device)
            
            # Forward pass
            outputs = self.model(image, demographic)
            loss = self.criterion(outputs, cognitive_scores)
            
            # Record metrics
            total_loss += loss.item()
            predictions.append(outputs.cpu().numpy())
            targets.append(cognitive_scores.cpu().numpy())
            
        # Calculate metrics
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        metrics = calculate_metrics(predictions, targets)
        metrics['loss'] = total_loss / len(data_loader)
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        save_dir: str
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
        """
        os.makedirs(save_dir, exist_ok=True)
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            print("Training metrics:", train_metrics)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            print("Validation metrics:", val_metrics)
            
            # Update learning rate
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                
                # Save model
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_metrics': val_metrics
                }
                torch.save(
                    checkpoint,
                    os.path.join(save_dir, 'best_model.pth')
                )
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                print("Early stopping triggered")
                break
                
        # Save final model
        torch.save(
            self.model.state_dict(),
            os.path.join(save_dir, 'final_model.pth')
        )
        
    def predict_longitudinal(
        self,
        data_loader: DataLoader
    ) -> pd.DataFrame:
        """
        Predict cognitive scores for longitudinal analysis.
        
        Args:
            data_loader: Data loader containing longitudinal data
            
        Returns:
            DataFrame with predictions and ground truth
        """
        self.model.eval()
        predictions = []
        targets = []
        ptids = []
        dates = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                # Move data to device
                image = batch['image'].to(self.device)
                demographic = batch['demographic'].to(self.device)
                cognitive_scores = batch['cognitive_scores']
                
                # Get predictions
                outputs = self.model(image, demographic)
                
                # Record results
                predictions.append(outputs.cpu().numpy())
                targets.append(cognitive_scores.numpy())
                ptids.extend(batch['ptid'])
                dates.extend(batch['mri_date'])
                
        # Combine results
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
        
        # Create DataFrame
        results = pd.DataFrame({
            'PTID': ptids,
            'mri_date': dates
        })
        
        # Add predictions and ground truth
        for i, feature in enumerate(self.config['model']['output_features']):
            results[f'pred_{feature}'] = predictions[:, i]
            results[f'true_{feature}'] = targets[:, i]
            
        return results 