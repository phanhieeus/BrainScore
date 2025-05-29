import os
import pandas as pd
import numpy as np

# Define the path to the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def normalize_scores(scores, score_ranges):
    normalized = {}
    for score_name, value in scores.items():
        min_val, max_val = score_ranges[score_name]
        normalized[score_name] = (value - min_val) / (max_val - min_val)
    return normalized

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

def normalize_test_pairs():
    # Load the test pairs data
    file_path = os.path.join(DATA_DIR, 'test_pairs.csv')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    df = pd.read_csv(file_path)
    
    # Define score ranges for normalization
    score_ranges = {
        'age': (50, 100),
        'PTEDUCAT': (5, 25),
        'ADAS11_now': (0, 70),
        'ADAS11_future': (0, 70),
        'ADAS13_now': (0, 85),
        'ADAS13_future': (0, 85),
        'MMSCORE_now': (0, 30),
        'MMSCORE_future': (0, 30)
    }
    
    # Normalize each column
    for col, (min_val, max_val) in score_ranges.items():
        df[col] = df[col].apply(lambda x: (x - min_val) / (max_val - min_val))
    
    # Save the normalized data
    output_path = os.path.join(DATA_DIR, 'test_pairs_normalized.csv')
    df.to_csv(output_path, index=False)
    print(f"Normalized data saved to {output_path}")
    print(f"Total rows: {len(df)}")
    
    # Save score ranges for later use in denormalization
    ranges_path = os.path.join(DATA_DIR, 'score_ranges.json')
    pd.Series(score_ranges).to_json(ranges_path)
    print(f"Score ranges saved to {ranges_path}")

def denormalize_predictions(predictions_df, score_ranges_path=None):
    """
    Convert normalized predictions back to original ranges.
    
    Args:
        predictions_df (pd.DataFrame): DataFrame containing normalized predictions
        score_ranges_path (str, optional): Path to score ranges JSON file. 
                                         If None, uses default ranges.
    
    Returns:
        pd.DataFrame: DataFrame with denormalized predictions
    """
    if score_ranges_path is None:
        score_ranges_path = os.path.join(DATA_DIR, 'score_ranges.json')
    
    if not os.path.exists(score_ranges_path):
        raise FileNotFoundError(f"Score ranges file not found at {score_ranges_path}")
    
    # Load score ranges
    score_ranges = pd.read_json(score_ranges_path).to_dict()
    
    # Create a copy of predictions to avoid modifying original
    denormalized_df = predictions_df.copy()
    
    # Denormalize each column that has a corresponding range
    for col in denormalized_df.columns:
        if col in score_ranges:
            min_val, max_val = score_ranges[col]
            denormalized_df[col] = denormalized_df[col].apply(
                lambda x: x * (max_val - min_val) + min_val
            )
    
    return denormalized_df

if __name__ == "__main__":
    normalize_test_pairs()
    
    # Example usage of denormalization:
    # normalized_predictions = pd.read_csv('predictions.csv')
    # denormalized_predictions = denormalize_predictions(normalized_predictions)
    # denormalized_predictions.to_csv('denormalized_predictions.csv', index=False) 