import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Define score ranges for denormalization
SCORE_RANGES = {
    'ADAS11': {'min': 0, 'max': 70},    # ADAS11 range: 0-70
    'ADAS13': {'min': 0, 'max': 85},    # ADAS13 range: 0-85
    'MMSCORE': {'min': 0, 'max': 30}    # MMSE range: 0-30
}

def denormalize_score(score, score_type):
    """
    Denormalize a normalized score back to its original scale
    
    Args:
        score (float): Normalized score
        score_type (str): Type of score ('ADAS11', 'ADAS13', or 'MMSCORE')
    """
    range_info = SCORE_RANGES[score_type]
    return score * (range_info['max'] - range_info['min']) + range_info['min']

def visualize_predictions(dataset_type):
    """
    Visualize predictions vs ground truth for future scores
    
    Args:
        dataset_type (str): Type of dataset ('train', 'val', or 'test')
    """
    # Define paths
    predictions_dir = os.path.join(os.path.dirname(__file__), '..', 'predictions')
    predictions_file = os.path.join(predictions_dir, f'{dataset_type}_predictions.csv')
    
    # Load predictions
    print(f"Loading predictions from {predictions_file}")
    df = pd.read_csv(predictions_file)
    
    # Define score columns
    score_columns = [
        'ADAS11_future', 'ADAS13_future', 'MMSCORE_future'
    ]
    
    # Create output directory for visualizations
    vis_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a figure with 1 row and 3 columns for future scores
    plt.figure(figsize=(20, 6))
    
    # Plot future scores
    for i, col in enumerate(score_columns):
        plt.subplot(1, 3, i + 1)
        
        # Get score type and denormalize
        score_type = col.split('_')[0]
        true_denorm = denormalize_score(df[f'{col}_true'], score_type)
        pred_denorm = denormalize_score(df[f'{col}_pred'], score_type)
        
        # Scatter plot of predictions vs ground truth
        plt.scatter(true_denorm, pred_denorm, alpha=0.5, label='Predictions')
        
        # Add perfect prediction line
        min_val = SCORE_RANGES[score_type]['min']
        max_val = SCORE_RANGES[score_type]['max']
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Set axis limits based on score range
        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        
        # Add labels and title
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title(f'{score_type} - {dataset_type.capitalize()} Set Predictions vs Ground Truth')
        plt.legend()
        
        # Add R² score
        r2 = np.corrcoef(true_denorm, pred_denorm)[0, 1] ** 2
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = os.path.join(vis_dir, f'{dataset_type}_predictions_vs_ground_truth.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_file}")
    
    # Create error distribution plots
    plt.figure(figsize=(20, 6))
    
    # Plot error distributions for future scores
    for i, col in enumerate(score_columns):
        plt.subplot(1, 3, i + 1)
        
        # Get score type and calculate denormalized errors
        score_type = col.split('_')[0]
        true_denorm = denormalize_score(df[f'{col}_true'], score_type)
        pred_denorm = denormalize_score(df[f'{col}_pred'], score_type)
        errors = pred_denorm - true_denorm
        
        # Plot error distribution
        sns.histplot(errors, kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title(f'{score_type} - {dataset_type.capitalize()} Set Error Distribution')
        
        # Add mean and std
        mean_error = errors.mean()
        std_error = errors.std()
        plt.text(0.05, 0.95, f'Mean = {mean_error:.3f}\nStd = {std_error:.3f}', 
                transform=plt.gca().transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = os.path.join(vis_dir, f'{dataset_type}_error_distributions.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved error distributions to {output_file}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Visualize predictions for BrainScore model')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'], required=True,
                      help='Dataset to visualize predictions for')
    args = parser.parse_args()
    
    visualize_predictions(args.dataset) 