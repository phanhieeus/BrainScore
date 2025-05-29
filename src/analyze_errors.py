import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def analyze_worst_predictions(dataset_type, n=10):
    """
    Analyze the n worst predictions based on MAE for each score type
    
    Args:
        dataset_type (str): Type of dataset ('train', 'val', or 'test')
        n (int): Number of worst predictions to analyze
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
    
    # Create output directory for analysis
    analysis_dir = os.path.join(os.path.dirname(__file__), '..', 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Analyze each score type
    for col in score_columns:
        print(f"\nAnalyzing {col}...")
        
        # Calculate absolute errors
        df[f'{col}_abs_error'] = df[f'{col}_error'].abs()
        
        # Get n worst predictions
        worst_predictions = df.nlargest(n, f'{col}_abs_error')
        
        # Create detailed analysis DataFrame
        analysis_df = worst_predictions[[
            'image_id', 'mri_date', 'EXAMDATE_now', 'EXAMDATE_future',
            'PTGENDER', 'age', 'PTEDUCAT', 'time_lapsed',
            f'{col}_pred', f'{col}_true', f'{col}_error', f'{col}_abs_error'
        ]].copy()
        
        # Save analysis to CSV
        output_file = os.path.join(analysis_dir, f'{dataset_type}_worst_{n}_{col}_predictions.csv')
        analysis_df.to_csv(output_file, index=False)
        print(f"Saved worst {n} predictions to {output_file}")
        
        # Print summary statistics
        print(f"\nWorst {n} predictions for {col}:")
        print(f"Mean Absolute Error: {analysis_df[f'{col}_abs_error'].mean():.2f}")
        print(f"Max Absolute Error: {analysis_df[f'{col}_abs_error'].max():.2f}")
        print(f"Mean Prediction: {analysis_df[f'{col}_pred'].mean():.2f}")
        print(f"Mean Ground Truth: {analysis_df[f'{col}_true'].mean():.2f}")
        
        # Create visualization
        plt.figure(figsize=(12, 6))
        
        # Plot predictions vs ground truth
        plt.subplot(1, 2, 1)
        plt.scatter(analysis_df[f'{col}_true'], analysis_df[f'{col}_pred'], alpha=0.6)
        plt.plot([analysis_df[f'{col}_true'].min(), analysis_df[f'{col}_true'].max()],
                [analysis_df[f'{col}_true'].min(), analysis_df[f'{col}_true'].max()],
                'r--', label='Perfect Prediction')
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.title(f'{col} - {dataset_type.capitalize()} Set Worst {n} Predictions')
        plt.legend()
        
        # Plot error distribution
        plt.subplot(1, 2, 2)
        sns.histplot(analysis_df[f'{col}_error'], kde=True)
        plt.xlabel('Prediction Error')
        plt.ylabel('Count')
        plt.title(f'{col} - {dataset_type.capitalize()} Set Error Distribution')
        
        # Save plot
        plt.tight_layout()
        plot_file = os.path.join(analysis_dir, f'{dataset_type}_worst_{n}_{col}_predictions.png')
        plt.savefig(plot_file)
        plt.close()
        print(f"Saved visualization to {plot_file}")
        
        # Print detailed information for each worst prediction
        print("\nDetailed information for each worst prediction:")
        for idx, row in analysis_df.iterrows():
            print(f"\nImage ID: {row['image_id']}")
            print(f"Age: {row['age']:.1f}, Education: {row['PTEDUCAT']:.1f}")
            print(f"Time between visits: {row['time_lapsed']:.1f} days")
            print(f"Prediction: {row[f'{col}_pred']:.2f}")
            print(f"Ground Truth: {row[f'{col}_true']:.2f}")
            print(f"Absolute Error: {row[f'{col}_abs_error']:.2f}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Analyze worst predictions for BrainScore model')
    parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'], required=True,
                      help='Dataset to analyze predictions for')
    parser.add_argument('--n', type=int, default=10,
                      help='Number of worst predictions to analyze (default: 10)')
    args = parser.parse_args()
    
    analyze_worst_predictions(args.dataset, args.n) 