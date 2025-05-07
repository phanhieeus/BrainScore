import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import GroupKFold
import yaml
from tqdm import tqdm

from data.dataset import BrainScoreDataset
from models.model import BrainScoreModel
from models.grad_cam import GradCAM
from training.trainer import Trainer
from utils.visualization import (
    plot_training_history,
    plot_predictions,
    plot_longitudinal_trajectories,
    plot_grad_cam_comparison
)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data(config: dict):
    """Prepare and split data."""
    # Load data
    cognitive_scores = pd.read_csv(config['data']['cognitive_scores'])
    demographics = pd.read_csv(config['data']['demographics'])
    
    # Create dataset
    dataset = BrainScoreDataset(
        image_dir=config['data']['image_dir'],
        cognitive_scores=cognitive_scores,
        demographics=demographics
    )
    
    # Split data by patient
    ptids = dataset.data['PTID'].unique()
    n_splits = int(1 / config['data']['train_val_test_split'][1])
    group_kfold = GroupKFold(n_splits=n_splits)
    
    # Get indices for each split
    train_idx, val_test_idx = next(group_kfold.split(
        dataset.data,
        groups=dataset.data['PTID']
    ))
    
    # Further split validation and test
    val_size = int(len(val_test_idx) * config['data']['train_val_test_split'][1] /
                  (config['data']['train_val_test_split'][1] + config['data']['train_val_test_split'][2]))
    val_idx = val_test_idx[:val_size]
    test_idx = val_test_idx[val_size:]
    
    # Create data loaders
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        torch.utils.data.Subset(dataset, test_idx),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Set device
    device = torch.device(config['training']['device'])
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    
    # Initialize trainer
    trainer = Trainer(config_path='configs/config.yaml', device=device)
    trainer.setup_model()
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['training']['num_epochs'],
        save_dir='results/models'
    )
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader)
    print("Test metrics:", test_metrics)
    
    # Get predictions for longitudinal analysis
    predictions = trainer.predict_longitudinal(test_loader)
    
    # Plot results
    plot_predictions(
        predictions=predictions,
        feature_names=config['model']['output_features'],
        save_dir='results/predictions'
    )
    
    plot_longitudinal_trajectories(
        predictions=predictions,
        feature_names=config['model']['output_features'],
        save_dir='results/trajectories'
    )
    
    # Generate Grad-CAM visualizations
    grad_cam = GradCAM(
        model=trainer.model,
        target_layer=config['grad_cam']['target_layer'],
        device=device
    )
    
    # Get a few samples for visualization
    for i, batch in enumerate(test_loader):
        if i >= config['evaluation']['visualization']['num_samples']:
            break
            
        image = batch['image'].to(device)
        demographic = batch['demographic'].to(device)
        
        # Generate Grad-CAM for each output
        for j, feature in enumerate(config['model']['output_features']):
            heatmap = grad_cam.generate_cam(
                image=image,
                demographic=demographic,
                target_score_idx=j
            )
            
            plot_grad_cam_comparison(
                original_image=image[0, 0].cpu().numpy(),
                heatmap=heatmap,
                feature_name=feature,
                save_dir='results/grad_cam'
            )

if __name__ == '__main__':
    main() 