import pandas as pd
import numpy as np
import random
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def split_data(input_file, train_file, val_file, test_file, train_ratio=0.8, val_ratio=0.1, random_state=42):
    """
    Split data into training, validation and test sets, ensuring:
    - No patients shared between sets
    - Maintain similar demographics distribution between sets
    - Default ratios: 80% train, 10% validation, 10% test
    - Input data should be normalized (from test_pairs_normalized.csv)
    """
    print("Reading normalized data...")
    df = pd.read_csv(input_file)
    
    # Convert date columns
    df['mri_date'] = pd.to_datetime(df['mri_date'])
    df['EXAMDATE_now'] = pd.to_datetime(df['EXAMDATE_now'])
    df['EXAMDATE_future'] = pd.to_datetime(df['EXAMDATE_future'])
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Get list of patients
    patients = df['PTID'].unique()
    n_patients = len(patients)
    n_val_patients = int(n_patients * val_ratio)
    n_test_patients = int(n_patients * val_ratio)  # Same as validation
    n_train_patients = n_patients - n_val_patients - n_test_patients
    
    print(f"\nTotal number of patients: {n_patients}")
    print(f"Number of patients for train set: {n_train_patients}")
    print(f"Number of patients for validation set: {n_val_patients}")
    print(f"Number of patients for test set: {n_test_patients}")
    
    # Function to calculate distribution statistics
    def get_distribution_stats(dataset):
        return {
            'gender_ratio': dataset['PTGENDER'].mean(),
            'age_mean': dataset['age'].mean(),
            'age_std': dataset['age'].std(),
            'educ_mean': dataset['PTEDUCAT'].mean(),
            'educ_std': dataset['PTEDUCAT'].std(),
            'time_lapsed_mean': dataset['time_lapsed'].mean(),
            'time_lapsed_std': dataset['time_lapsed'].std()
        }
    
    # Function to calculate distribution difference
    def calculate_distribution_diff(stats1, stats2):
        return {
            'gender_diff': abs(stats1['gender_ratio'] - stats2['gender_ratio']),
            'age_diff': abs(stats1['age_mean'] - stats2['age_mean']),
            'educ_diff': abs(stats1['educ_mean'] - stats2['educ_mean']),
            'time_lapsed_diff': abs(stats1['time_lapsed_mean'] - stats2['time_lapsed_mean'])
        }
    
    # Function to check if split is representative
    def is_representative(val_stats, test_stats, threshold=0.1):
        diff = calculate_distribution_diff(val_stats, test_stats)
        # Check if all differences are below threshold
        return all(v < threshold for v in diff.values())
    
    # Try multiple splits to find the most balanced one
    best_diff = float('inf')
    best_splits = None
    max_attempts = 1000  # Increase number of attempts to find better split
    
    print("\nFinding balanced split...")
    for attempt in range(max_attempts):
        # Randomly select patients for validation and test sets
        remaining_patients = list(patients)
        random.shuffle(remaining_patients)
        
        val_patients = set(remaining_patients[:n_val_patients])
        test_patients = set(remaining_patients[n_val_patients:n_val_patients + n_test_patients])
        train_patients = set(remaining_patients[n_val_patients + n_test_patients:])
        
        # Create temporary sets
        temp_val_df = df[df['PTID'].isin(val_patients)].copy()
        temp_test_df = df[df['PTID'].isin(test_patients)].copy()
        
        # Calculate distribution statistics
        val_stats = get_distribution_stats(temp_val_df)
        test_stats = get_distribution_stats(temp_test_df)
        
        # Check if split is representative
        if is_representative(val_stats, test_stats):
            # Calculate distribution difference
            diff = calculate_distribution_diff(val_stats, test_stats)
            total_diff = sum(diff.values())
            
            if total_diff < best_diff:
                best_diff = total_diff
                best_splits = (train_patients, val_patients, test_patients)
                print(f"Found better split (attempt {attempt + 1}): {total_diff:.4f}")
    
    if best_splits is None:
        print("Warning: Could not find a fully representative split. Using the best available split.")
        # Use the last split as it's the best we found
        best_splits = (train_patients, val_patients, test_patients)
    
    # Use the best split found
    train_patients, val_patients, test_patients = best_splits
    
    # Create final sets
    train_df = df[df['PTID'].isin(train_patients)].copy()
    val_df = df[df['PTID'].isin(val_patients)].copy()
    test_df = df[df['PTID'].isin(test_patients)].copy()
    
    # Save results
    print(f"\nSaving results...")
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)
    
    # Print statistics
    print("\n=== STATISTICS ===\n")
    print(f"Total data points: {len(df)}")
    print(f"Train set: {len(train_df)} points ({len(train_patients)} patients)")
    print(f"Validation set: {len(val_df)} points ({len(val_patients)} patients)")
    print(f"Test set: {len(test_df)} points ({len(test_patients)} patients)")
    
    # Analyze demographics
    print("\nDemographics analysis:")
    for dataset_name, dataset in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        print(f"\n{dataset_name} set:")
        
        print("\nGender:")
        gender_dist = dataset['PTGENDER'].value_counts()
        for gender, count in gender_dist.items():
            percentage = (count / len(dataset)) * 100
            gender_name = "Male" if gender == 1 else "Female"
            print(f"- {gender_name}: {count} points ({percentage:.1f}%)")
        
        print("\nAge (normalized):")
        print(f"- Average: {dataset['age'].mean():.3f}")
        print(f"- Minimum: {dataset['age'].min():.3f}")
        print(f"- Maximum: {dataset['age'].max():.3f}")
        
        print("\nEducation (normalized):")
        print(f"- Average: {dataset['PTEDUCAT'].mean():.3f}")
        print(f"- Minimum: {dataset['PTEDUCAT'].min():.3f}")
        print(f"- Maximum: {dataset['PTEDUCAT'].max():.3f}")
        
        print("\nTime between tests (days):")
        print(f"- Average: {dataset['time_lapsed'].mean():.1f}")
        print(f"- Minimum: {dataset['time_lapsed'].min():.1f}")
        print(f"- Maximum: {dataset['time_lapsed'].max():.1f}")
    
    # Print distribution differences
    print("\nDistribution differences between validation and test sets:")
    val_stats = get_distribution_stats(val_df)
    test_stats = get_distribution_stats(test_df)
    diff = calculate_distribution_diff(val_stats, test_stats)
    print(f"- Gender ratio difference: {diff['gender_diff']:.3f}")
    print(f"- Age mean difference: {diff['age_diff']:.3f}")
    print(f"- Education mean difference: {diff['educ_diff']:.3f}")
    print(f"- Time lapsed difference: {diff['time_lapsed_diff']:.1f} days")

if __name__ == "__main__":
    input_file = "data/test_pairs_normalized.csv"  # Updated to use normalized data
    train_file = "data/train_data.csv"
    val_file = "data/val_data.csv"
    test_file = "data/test_data.csv"
    split_data(input_file, train_file, val_file, test_file) 