import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

def create_test_pairs(cognitive_scores_file, demographics_file, mri_dir, output_file):
    """
    Create dataset with test pairs where:
    - Split data into MRI data and cognitive test data
    - Filter out MRIs without corresponding image files
    - For each patient, create pairs of tests:
        + One test within 30 days of MRI
        + One test between 180-360 days after MRI
    - Combine with demographics data
    """
    print("Reading data...")
    cognitive_scores_df = pd.read_csv(cognitive_scores_file, delimiter=',')
    demographics = pd.read_csv(demographics_file, delimiter=',')
    
    # Convert date columns
    cognitive_scores_df['EXAMDATE'] = pd.to_datetime(cognitive_scores_df['EXAMDATE'])
    cognitive_scores_df['mri_date'] = pd.to_datetime(cognitive_scores_df['mri_date'])
    
    # Remove rows with negative cognitive test scores
    initial_rows = len(cognitive_scores_df)
    cognitive_scores_df = cognitive_scores_df[
        (cognitive_scores_df['ADAS11'] >= 0) & 
        (cognitive_scores_df['ADAS13'] >= 0) & 
        (cognitive_scores_df['MMSCORE'] >= 0)
    ]
    removed_rows = initial_rows - len(cognitive_scores_df)
    if removed_rows > 0:
        print(f"\nRemoved {removed_rows} rows with negative cognitive test scores")
    
    # Process demographics data
    demographics['PTGENDER'] = (demographics['PTGENDER'] == 1.0).astype(int)  # 1.0 for male, 2.0 for female
    demographics['PTDOBYY'] = pd.to_numeric(demographics['PTDOBYY'], errors='coerce')
    demographics['PTEDUCAT'] = pd.to_numeric(demographics['PTEDUCAT'], errors='coerce')
    
    # Split data into MRI data and cognitive test data
    mri_data = cognitive_scores_df[['PTID', 'mri_date', 'image_id']].drop_duplicates()
    test_data = cognitive_scores_df[['PTID', 'EXAMDATE', 'ADAS11', 'ADAS13', 'MMSCORE']].drop_duplicates()
    
    # Check for corresponding MRI image files
    print("\nChecking MRI image files...")
    valid_mri_ids = set()
    for mri_id in mri_data['image_id'].unique():
        mri_path = os.path.join(mri_dir, f"I{mri_id}", "T1_biascorr_brain.nii.gz")
        if os.path.exists(mri_path):
            valid_mri_ids.add(mri_id)
        else:
            print(f"Warning: Image file not found for mri_id {mri_id}")
    
    # Filter mri_data to keep only valid mri_ids
    mri_data = mri_data[mri_data['image_id'].isin(valid_mri_ids)]
    print(f"\nNumber of valid mri_ids: {len(valid_mri_ids)}")
    
    # Create DataFrame to store results
    result_rows = []
    
    # Get list of patients
    patients = mri_data['PTID'].unique()
    total_patients = len(patients)
    
    print(f"\nProcessing data for {total_patients} patients...")
    
    # Process each patient
    for i, patient in enumerate(patients, 1):
        # Get patient data
        patient_mri = mri_data[mri_data['PTID'] == patient]
        patient_tests = test_data[test_data['PTID'] == patient]
        patient_demo = demographics[demographics['PTID'] == patient]
        
        if len(patient_demo) == 0:
            print(f"No demographics data found for patient {patient}")
            continue
            
        patient_demo = patient_demo.iloc[0]
        
        # Process each MRI
        for _, mri_row in patient_mri.iterrows():
            mri_date = mri_row['mri_date']
            image_id = mri_row['image_id']
            
            # Find tests within 30 days of MRI
            near_tests = patient_tests[
                (patient_tests['EXAMDATE'] - mri_date).dt.days.between(-30, 30)
            ]
            
            # Get all tests after near tests
            future_tests = patient_tests[
                patient_tests['EXAMDATE'] > mri_date
            ]
            
            # Create pairs of tests
            for _, near_test in near_tests.iterrows():
                for _, future_test in future_tests.iterrows():
                    # Check if time between tests is between 180-540 days
                    time_between_tests = (future_test['EXAMDATE'] - near_test['EXAMDATE']).days
                    if not (180 <= time_between_tests <= 540):
                        continue
                        
                    # Calculate age at MRI time
                    age = mri_date.year - patient_demo['PTDOBYY']
                    
                    # Create new data point
                    data_point = {
                        'PTID': patient,
                        'mri_date': mri_date,
                        'image_id': image_id,
                        'EXAMDATE_now': near_test['EXAMDATE'],
                        'EXAMDATE_future': future_test['EXAMDATE'],
                        'PTGENDER': patient_demo['PTGENDER'],
                        'age': age,
                        'PTEDUCAT': patient_demo['PTEDUCAT'],
                        'time_lapsed': (future_test['EXAMDATE'] - near_test['EXAMDATE']).days,
                        'ADAS11_now': near_test['ADAS11'],
                        'ADAS13_now': near_test['ADAS13'],
                        'MMSCORE_now': near_test['MMSCORE'],
                        'ADAS11_future': future_test['ADAS11'],
                        'ADAS13_future': future_test['ADAS13'],
                        'MMSCORE_future': future_test['MMSCORE']
                    }
                    result_rows.append(data_point)
        
    
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
    
    print("\nTime between tests:")
    print(f"- Average: {result_df['time_lapsed'].mean():.1f} days")
    print(f"- Minimum: {result_df['time_lapsed'].min():.1f} days")
    print(f"- Maximum: {result_df['time_lapsed'].max():.1f} days")

if __name__ == "__main__":
    cognitive_scores_file = "data/c1_c2_cognitive_score.csv"
    demographics_file = "data/c1_c2_demographics.csv"
    mri_dir = "data/T1_biascorr_brain_data"
    output_file = "data/test_pairs.csv"
    create_test_pairs(cognitive_scores_file, demographics_file, mri_dir, output_file) 