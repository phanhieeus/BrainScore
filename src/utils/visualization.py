import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime

def plot_training_history(
    train_metrics: Dict[str, List[float]],
    val_metrics: Dict[str, List[float]],
    save_dir: str
):
    """
    Plot training history.
    
    Args:
        train_metrics: Dictionary of training metrics over epochs
        val_metrics: Dictionary of validation metrics over epochs
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['loss'], label='Training Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_history.png'))
    plt.close()
    
    # Plot metrics
    metrics = ['mae_avg', 'rmse_avg', 'r2_avg']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(train_metrics[metric], label=f'Training {metric}')
        plt.plot(val_metrics[metric], label=f'Validation {metric}')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Training and Validation {metric}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{metric}_history.png'))
        plt.close()

def plot_predictions(
    predictions: pd.DataFrame,
    feature_names: List[str],
    save_dir: str
):
    """
    Plot scatter plots of predictions vs ground truth.
    
    Args:
        predictions: DataFrame with predictions and ground truth
        feature_names: List of feature names
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for feature in feature_names:
        plt.figure(figsize=(10, 6))
        plt.scatter(
            predictions[f'true_{feature}'],
            predictions[f'pred_{feature}'],
            alpha=0.5
        )
        
        # Add diagonal line
        min_val = min(
            predictions[f'true_{feature}'].min(),
            predictions[f'pred_{feature}'].min()
        )
        max_val = max(
            predictions[f'true_{feature}'].max(),
            predictions[f'pred_{feature}'].max()
        )
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel(f'True {feature}')
        plt.ylabel(f'Predicted {feature}')
        plt.title(f'Predictions vs Ground Truth - {feature}')
        plt.savefig(os.path.join(save_dir, f'predictions_{feature}.png'))
        plt.close()

def plot_longitudinal_trajectories(
    predictions: pd.DataFrame,
    feature_names: List[str],
    save_dir: str
):
    """
    Plot longitudinal trajectories of cognitive scores.
    
    Args:
        predictions: DataFrame with predictions and ground truth over time
        feature_names: List of feature names
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert dates to datetime
    predictions['mri_date'] = pd.to_datetime(predictions['mri_date'])
    
    # Plot for each patient
    for ptid in predictions['PTID'].unique():
        patient_data = predictions[predictions['PTID'] == ptid]
        
        for feature in feature_names:
            plt.figure(figsize=(10, 6))
            
            # Plot true values
            plt.plot(
                patient_data['mri_date'],
                patient_data[f'true_{feature}'],
                'bo-',
                label='True'
            )
            
            # Plot predictions
            plt.plot(
                patient_data['mri_date'],
                patient_data[f'pred_{feature}'],
                'ro--',
                label='Predicted'
            )
            
            plt.xlabel('Date')
            plt.ylabel(feature)
            plt.title(f'Longitudinal Trajectory - {feature} (Patient {ptid})')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(
                os.path.join(save_dir, f'trajectory_{ptid}_{feature}.png')
            )
            plt.close()

def plot_feature_importance(
    importance_scores: Dict[str, float],
    save_dir: str
):
    """
    Plot feature importance scores.
    
    Args:
        importance_scores: Dictionary of feature importance scores
        save_dir: Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    features = list(importance_scores.keys())
    scores = list(importance_scores.values())
    
    # Sort by importance
    sorted_idx = np.argsort(scores)
    features = [features[i] for i in sorted_idx]
    scores = [scores[i] for i in sorted_idx]
    
    plt.barh(range(len(features)), scores)
    plt.yticks(range(len(features)), features)
    plt.xlabel('Importance Score')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'))
    plt.close()

def plot_grad_cam_comparison(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    feature_name: str,
    save_dir: str
):
    """
    Plot comparison of original image and Grad-CAM heatmap.
    
    Args:
        original_image: Original MRI image
        heatmap: Grad-CAM heatmap
        feature_name: Name of the predicted feature
        save_dir: Directory to save plot
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot middle slices
    middle_slices = [s // 2 for s in original_image.shape]
    
    for i, (ax, slice_idx) in enumerate(zip(axes, middle_slices)):
        if i == 0:  # Axial
            img_slice = original_image[slice_idx, :, :]
            heat_slice = heatmap[slice_idx, :, :]
        elif i == 1:  # Coronal
            img_slice = original_image[:, slice_idx, :]
            heat_slice = heatmap[:, slice_idx, :]
        else:  # Sagittal
            img_slice = original_image[:, :, slice_idx]
            heat_slice = heatmap[:, :, slice_idx]
            
        # Plot original image
        ax.imshow(img_slice, cmap='gray')
        
        # Overlay heatmap
        ax.imshow(heat_slice, cmap='jet', alpha=0.5)
        ax.axis('off')
        
    plt.suptitle(f'Grad-CAM Visualization - {feature_name}')
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f'grad_cam_{feature_name}.png')
    )
    plt.close() 