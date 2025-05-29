import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def visualize_predictions(dataset_type):
    """
    Visualize all predictions vs ground truth for each score type
    
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
        'ADAS11_now', 'ADAS13_now', 'MMSCORE_now',
        'ADAS11_future', 'ADAS13_future', 'MMSCORE_future'
    ]
    
    # Create output directory for visualizations
    vis_dir = os.path.join(os.path.dirname(__file__), '..', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create a figure with 2 rows and 3 columns for current scores
    plt.figure(figsize=(20, 12))
    
    # Plot current scores
    for i, col in enumerate(['ADAS11_now', 'ADAS13_now', 'MMSCORE_now']):
        plt.subplot(2, 3, i + 1)
        
        # Scatter plot of predictions vs ground truth
        plt.scatter(df[f'{col}_true'], df[f'{col}_pred'], alpha=0.5, label='Predictions')
        
        # Add perfect prediction line
        min_val = min(df[f'{col}_true'].min(), df[f'{col}_pred'].min())
        max_val = max(df[f'{col}_true'].max(), df[f'{col}_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add labels and title
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title(f'{col} - {dataset_type.capitalize()} Set Predictions vs Ground Truth')
        plt.legend()
        
        # Add R² score
        r2 = np.corrcoef(df[f'{col}_true'], df[f'{col}_pred'])[0, 1] ** 2
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)
    
    # Plot future scores
    for i, col in enumerate(['ADAS11_future', 'ADAS13_future', 'MMSCORE_future']):
        plt.subplot(2, 3, i + 4)
        
        # Scatter plot of predictions vs ground truth
        plt.scatter(df[f'{col}_true'], df[f'{col}_pred'], alpha=0.5, label='Predictions')
        
        # Add perfect prediction line
        min_val = min(df[f'{col}_true'].min(), df[f'{col}_pred'].min())
        max_val = max(df[f'{col}_true'].max(), df[f'{col}_pred'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        # Add labels and title
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title(f'{col} - {dataset_type.capitalize()} Set Predictions vs Ground Truth')
        plt.legend()
        
        # Add R² score
        r2 = np.corrcoef(df[f'{col}_true'], df[f'{col}_pred'])[0, 1] ** 2
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes)
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = os.path.join(vis_dir, f'{dataset_type}_predictions_vs_ground_truth.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_file}")
    
    # Create error distribution plots
    plt.figure(figsize=(20, 12))
    
    # Plot error distributions for current scores
    for i, col in enumerate(['ADAS11_now', 'ADAS13_now', 'MMSCORE_now']):
        plt.subplot(2, 3, i + 1)
        
        # Plot error distribution
        sns.histplot(df[f'{col}_error'], kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title(f'{col} - {dataset_type.capitalize()} Set Error Distribution')
        
        # Add mean and std
        mean_error = df[f'{col}_error'].mean()
        std_error = df[f'{col}_error'].std()
        plt.text(0.05, 0.95, f'Mean = {mean_error:.3f}\nStd = {std_error:.3f}', 
                transform=plt.gca().transAxes)
    
    # Plot error distributions for future scores
    for i, col in enumerate(['ADAS11_future', 'ADAS13_future', 'MMSCORE_future']):
        plt.subplot(2, 3, i + 4)
        
        # Plot error distribution
        sns.histplot(df[f'{col}_error'], kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title(f'{col} - {dataset_type.capitalize()} Set Error Distribution')
        
        # Add mean and std
        mean_error = df[f'{col}_error'].mean()
        std_error = df[f'{col}_error'].std()
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