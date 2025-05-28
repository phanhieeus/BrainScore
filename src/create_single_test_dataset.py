import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

def create_single_test_dataset(input_file, demographics_file, mri_dir, output_file):
    """
    Create dataset with single test points, where:
    - For each patient, combine all MRIs with all tests
    - Keep only pairs where EXAMDATE is after mri_date
    - Combine with demographics data
    - Keep only mri_ids with corresponding image files in T1_biascorr_brain_data directory
    """
    print("Reading data...")
    df = pd.read_csv(input_file, delimiter=',')
    demographics = pd.read_csv(demographics_file)
    
    # Convert date columns
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
    df['mri_date'] = pd.to_datetime(df['mri_date'])
    
    # Process demographics data
    demographics['PTGENDER'] = (demographics['PTGENDER'] == 1.0).astype(int)  # 1.0 for male, 2.0 for female
    demographics['PTDOBYY'] = pd.to_numeric(demographics['PTDOBYY'], errors='coerce')
    demographics['PTEDUCAT'] = pd.to_numeric(demographics['PTEDUCAT'], errors='coerce')
    
    # Check for corresponding MRI image files
    print("\nChecking MRI image files...")
    valid_mri_ids = set()
    for mri_id in df['image_id'].unique():
        mri_path = os.path.join(mri_dir, f"I{mri_id}", "T1_biascorr_brain.nii.gz")
        if os.path.exists(mri_path):
            valid_mri_ids.add(mri_id)
        else:
            print(f"Warning: Image file not found for mri_id {mri_id}")
    
    # Filter data to keep only valid mri_ids
    df = df[df['image_id'].isin(valid_mri_ids)]
    print(f"\nNumber of valid mri_ids: {len(valid_mri_ids)}")
    
    # Create DataFrame to store results
    result_rows = []
    
    # Get list of patients
    patients = df['PTID'].unique()
    total_patients = len(patients)
    
    print(f"\nProcessing data for {total_patients} patients...")
    
    # Process each patient
    for i, patient in enumerate(patients, 1):
        # Get patient data
        patient_data = df[df['PTID'] == patient]
        patient_demo = demographics[demographics['PTID'] == patient]
        
        if len(patient_demo) == 0:
            print(f"No demographics data found for patient {patient}")
            continue
            
        patient_demo = patient_demo.iloc[0]
        
        # Get list of MRI dates
        mri_dates = patient_data['mri_date'].unique()
        
        # Get list of test dates
        test_dates = patient_data['EXAMDATE'].unique()
        
        # Combine each MRI with each test
        for mri_date in mri_dates:
            for test_date in test_dates:
                # Calculate time difference in days
                time_diff = (test_date - mri_date).days
                
                # Keep only pairs where test_date is after mri_date
                if time_diff >= 0:
                    # Get corresponding test data
                    test_data = patient_data[patient_data['EXAMDATE'] == test_date].iloc[0]
                    
                    # Calculate age at MRI time
                    age = mri_date.year - patient_demo['PTDOBYY']
                    
                    # Create new data point
                    data_point = {
                        'PTID': patient,
                        'image_id': test_data['image_id'],
                        'mri_date': mri_date,
                        'EXAMDATE': test_date,
                        'test_mri_time_diff': time_diff,
                        'PTGENDER': patient_demo['PTGENDER'],
                        'age': age,
                        'PTEDUCAT': patient_demo['PTEDUCAT'],
                        'ADAS11': test_data['ADAS11'],
                        'ADAS13': test_data['ADAS13'],
                        'MMSCORE': test_data['MMSCORE'],
                        'CDGLOBAL': test_data['CDGLOBAL']
                    }
                    result_rows.append(data_point)
        
        # Print progress
        if i % 50 == 0 or i == total_patients:
            print(f"Processed {i}/{total_patients} patients")
    
    # Create result DataFrame
    result_df = pd.DataFrame(result_rows)
    
    # Sort results by PTID and mri_date
    result_df = result_df.sort_values(['PTID', 'mri_date'])
    
    # Save results
    print(f"\nSaving results to {output_file}...")
    result_df.to_csv(output_file, index=False)
    
    # Print statistics
    print("\n=== STATISTICS ===\n")
    print(f"Total data points: {len(result_df)}")
    print(f"Number of patients: {result_df['PTID'].nunique()}")
    print(f"Number of valid mri_ids: {len(valid_mri_ids)}")
    
    # Analyze time distribution
    print("\nTime distribution:")
    bins = [0, 180, 365, 547, 730, 912, 1095, 1277, 1460, 1642, 1825, float('inf')]
    labels = [
        '0-6 months', '6-12 months', '12-18 months', '18-24 months', 
        '24-30 months', '30-36 months', '36-42 months', '42-48 months',
        '48-54 months', '54-60 months', '> 60 months'
    ]
    
    # Analyze time distribution
    result_df['time_range'] = pd.cut(result_df['test_mri_time_diff'], bins=bins, labels=labels)
    time_dist = result_df['time_range'].value_counts().sort_index()
    
    for time_range, count in time_dist.items():
        percentage = (count / len(result_df)) * 100
        print(f"- {time_range}: {count} points ({percentage:.1f}%)")
    
    # Analyze demographics data
    print("\nDemographics analysis:")
    print("\nGender:")
    gender_dist = result_df['PTGENDER'].value_counts()
    for gender, count in gender_dist.items():
        percentage = (count / len(result_df)) * 100
        gender_name = "Male" if gender == 1 else "Female"
        print(f"- {gender_name}: {count} points ({percentage:.1f}%)")
    
    print("\nAge:")
    print(f"- Average: {result_df['age'].mean():.1f} years")
    print(f"- Minimum: {result_df['age'].min():.1f} years")
    print(f"- Maximum: {result_df['age'].max():.1f} years")
    
    print("\nEducation:")
    print(f"- Average: {result_df['PTEDUCAT'].mean():.1f} years")
    print(f"- Minimum: {result_df['PTEDUCAT'].min():.1f} years")
    print(f"- Maximum: {result_df['PTEDUCAT'].max():.1f} years")

if __name__ == "__main__":
    input_file = "data/c1_c2_cognitive_score.csv"
    demographics_file = "data/c1_c2_demographics.csv"
    mri_dir = "data/T1_biascorr_brain_data"
    output_file = "data/single_test_points.csv"
    create_single_test_dataset(input_file, demographics_file, mri_dir, output_file) 