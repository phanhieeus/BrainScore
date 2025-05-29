import os
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def denormalize_scores(normalized_scores, score_ranges):
    """
    Convert normalized scores back to original ranges.
    
    Args:
        normalized_scores (dict): Dictionary of normalized scores
        score_ranges (dict): Dictionary of (min, max) ranges for each score
        
    Returns:
        dict: Dictionary of denormalized scores
    """
    denormalized = {}
    for score_name, normalized_value in normalized_scores.items():
        min_val, max_val = score_ranges[score_name]
        denormalized[score_name] = normalized_value * (max_val - min_val) + min_val
    return denormalized

def main():
    # Define paths
    predictions_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'predictions')
    predictions_file = os.path.join(predictions_dir, 'test_predictions.csv')
    
    # Define score ranges for denormalization
    score_ranges = {
        'ADAS11_now': (0, 70),
        'ADAS11_future': (0, 70),
        'ADAS13_now': (0, 85),
        'ADAS13_future': (0, 85),
        'MMSCORE_now': (0, 30),
        'MMSCORE_future': (0, 30)
    }
    
    # Load predictions
    print(f"Loading predictions from {predictions_file}")
    predictions_df = pd.read_csv(predictions_file)
    
    # Create a copy of predictions for denormalization
    denormalized_df = predictions_df.copy()
    
    # Denormalize each score column
    for col, (min_val, max_val) in score_ranges.items():
        # Denormalize predictions
        if f'{col}_pred' in denormalized_df.columns:
            denormalized_df[f'{col}_pred'] = denormalized_df[f'{col}_pred'] * (max_val - min_val) + min_val
        
        # Denormalize ground truth
        if f'{col}_true' in denormalized_df.columns:
            denormalized_df[f'{col}_true'] = denormalized_df[f'{col}_true'] * (max_val - min_val) + min_val
        
        # Recalculate errors
        if f'{col}_error' in denormalized_df.columns:
            denormalized_df[f'{col}_error'] = (
                denormalized_df[f'{col}_pred'] - 
                denormalized_df[f'{col}_true']
            )
    
    # Save denormalized predictions
    output_file = os.path.join(predictions_dir, 'test_predictions_denormalized.csv')
    denormalized_df.to_csv(output_file, index=False)
    print(f"Denormalized predictions saved to {output_file}")
    
    # Print detailed statistics
    print("\nPrediction Statistics:")
    for col in score_ranges.keys():
        print(f"\n{col}:")
        print(f"Mean Prediction: {denormalized_df[f'{col}_pred'].mean():.2f}")
        print(f"Mean Ground Truth: {denormalized_df[f'{col}_true'].mean():.2f}")
        print(f"Mean Absolute Error: {denormalized_df[f'{col}_error'].abs().mean():.2f}")
        print(f"Root Mean Square Error: {np.sqrt((denormalized_df[f'{col}_error']**2).mean()):.2f}")
        print(f"R2 Score: {r2_score(denormalized_df[f'{col}_true'], denormalized_df[f'{col}_pred']):.4f}")
        
        # Print min/max values
        print(f"Prediction Range: [{denormalized_df[f'{col}_pred'].min():.2f}, {denormalized_df[f'{col}_pred'].max():.2f}]")
        print(f"Ground Truth Range: [{denormalized_df[f'{col}_true'].min():.2f}, {denormalized_df[f'{col}_true'].max():.2f}]")
        
        # Print error statistics
        print(f"Error Range: [{denormalized_df[f'{col}_error'].min():.2f}, {denormalized_df[f'{col}_error'].max():.2f}]")
        print(f"Error Std Dev: {denormalized_df[f'{col}_error'].std():.2f}")

if __name__ == "__main__":
    main() 