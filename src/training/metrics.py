import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple

def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for regression tasks.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Calculate metrics for each output
    for i in range(predictions.shape[1]):
        pred = predictions[:, i]
        target = targets[:, i]
        
        # Mean Absolute Error
        mae = mean_absolute_error(target, pred)
        metrics[f'mae_{i}'] = mae
        
        # Root Mean Squared Error
        rmse = np.sqrt(mean_squared_error(target, pred))
        metrics[f'rmse_{i}'] = rmse
        
        # R-squared
        r2 = r2_score(target, pred)
        metrics[f'r2_{i}'] = r2
        
    # Calculate average metrics
    metrics['mae_avg'] = np.mean([metrics[f'mae_{i}'] for i in range(predictions.shape[1])])
    metrics['rmse_avg'] = np.mean([metrics[f'rmse_{i}'] for i in range(predictions.shape[1])])
    metrics['r2_avg'] = np.mean([metrics[f'r2_{i}'] for i in range(predictions.shape[1])])
    
    return metrics

def calculate_longitudinal_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    time_points: np.ndarray
) -> Dict[str, float]:
    """
    Calculate metrics for longitudinal analysis.
    
    Args:
        predictions: Model predictions over time
        targets: Ground truth values over time
        time_points: Time points for each prediction
        
    Returns:
        Dictionary of longitudinal metrics
    """
    metrics = {}
    
    # Calculate rate of change
    for i in range(predictions.shape[1]):
        pred = predictions[:, i]
        target = targets[:, i]
        
        # Calculate slopes
        pred_slope = np.polyfit(time_points, pred, 1)[0]
        target_slope = np.polyfit(time_points, target, 1)[0]
        
        # Calculate correlation between predicted and actual slopes
        slope_corr = np.corrcoef(pred_slope, target_slope)[0, 1]
        metrics[f'slope_corr_{i}'] = slope_corr
        
        # Calculate mean absolute error of slopes
        slope_mae = np.abs(pred_slope - target_slope)
        metrics[f'slope_mae_{i}'] = slope_mae
        
    # Calculate average metrics
    metrics['slope_corr_avg'] = np.mean([metrics[f'slope_corr_{i}'] for i in range(predictions.shape[1])])
    metrics['slope_mae_avg'] = np.mean([metrics[f'slope_mae_{i}'] for i in range(predictions.shape[1])])
    
    return metrics 