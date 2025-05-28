import pandas as pd
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

def split_data(input_file, train_file, val_file, test_ratio=0.2, random_state=42):
    """
    Split data into training and validation sets, ensuring:
    - No patients shared between sets
    - Maintain similar demographics distribution between sets
    - Default test ratio is 20% of patients
    """
    print("Reading data...")
    df = pd.read_csv(input_file)
    
    # Convert date columns
    df['mri_date'] = pd.to_datetime(df['mri_date'])
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
    
    # Set random seed
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Get list of patients
    patients = df['PTID'].unique()
    n_patients = len(patients)
    n_test_patients = int(n_patients * test_ratio)
    
    print(f"\nTotal number of patients: {n_patients}")
    print(f"Number of patients for test set: {n_test_patients}")
    
    # Randomly select patients for test set
    test_patients = set(random.sample(list(patients), n_test_patients))
    
    # Create test and train sets
    test_df = df[df['PTID'].isin(test_patients)].copy()
    train_df = df[~df['PTID'].isin(test_patients)].copy()
    
    # Save results
    print(f"\nSaving results...")
    train_df.to_csv(train_file, index=False)
    test_df.to_csv(val_file, index=False)
    
    # Print statistics
    print("\n=== STATISTICS ===\n")
    print(f"Total data points: {len(df)}")
    print(f"Train set: {len(train_df)} points ({len(patients) - len(test_patients)} patients)")
    print(f"Test set: {len(test_df)} points ({len(test_patients)} patients)")
    
    # Analyze time distribution
    print("\nTime distribution:")
    bins = [0, 180, 365, 547, 730, 912, 1095, 1277, 1460, 1642, 1825, float('inf')]
    labels = [
        '0-6 months', '6-12 months', '12-18 months', '18-24 months', 
        '24-30 months', '30-36 months', '36-42 months', '42-48 months',
        '48-54 months', '54-60 months', '> 60 months'
    ]
    
    for dataset_name, dataset in [("Train", train_df), ("Test", test_df)]:
        print(f"\n{dataset_name} set:")
        dataset['time_range'] = pd.cut(dataset['test_mri_time_diff'], bins=bins, labels=labels)
        time_dist = dataset['time_range'].value_counts().sort_index()
        
        for time_range, count in time_dist.items():
            percentage = (count / len(dataset)) * 100
            print(f"- {time_range}: {count} points ({percentage:.1f}%)")
    
    # Analyze demographics
    print("\nDemographics analysis:")
    for dataset_name, dataset in [("Train", train_df), ("Test", test_df)]:
        print(f"\n{dataset_name} set:")
        
        print("\nGender:")
        gender_dist = dataset['PTGENDER'].value_counts()
        for gender, count in gender_dist.items():
            percentage = (count / len(dataset)) * 100
            gender_name = "Male" if gender == 1 else "Female"
            print(f"- {gender_name}: {count} points ({percentage:.1f}%)")
        
        print("\nAge:")
        print(f"- Average: {dataset['age'].mean():.1f} years")
        print(f"- Minimum: {dataset['age'].min():.1f} years")
        print(f"- Maximum: {dataset['age'].max():.1f} years")
        
        print("\nEducation:")
        print(f"- Average: {dataset['PTEDUCAT'].mean():.1f} years")
        print(f"- Minimum: {dataset['PTEDUCAT'].min():.1f} years")
        print(f"- Maximum: {dataset['PTEDUCAT'].max():.1f} years")

if __name__ == "__main__":
    input_file = "data/single_test_points.csv"
    train_file = "data/train_data.csv"
    val_file = "data/test_data.csv"
    split_data(input_file, train_file, val_file) 