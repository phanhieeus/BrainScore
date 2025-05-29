import pandas as pd
import os
from datetime import datetime

def compare_data_files(file1_path, file2_path):
    """
    Compare two CSV files and find rows that don't match based on specific columns.
    Only compare PTID, mri_date, image_id, and EXAMDATE_now/EXAMDATE columns.
    
    Args:
        file1_path (str): Path to first CSV file (test_pairs.csv)
        file2_path (str): Path to second CSV file (data_followup_6m_18m_1m.csv)
    """
    print("Reading data files...")
    # Read CSV files
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # Print initial info
    print("\nInitial data info:")
    print(f"Columns in {os.path.basename(file1_path)}:", df1.columns.tolist())
    print(f"Columns in {os.path.basename(file2_path)}:", df2.columns.tolist())
    
    # Convert date columns to datetime
    date_columns = {
        'test_pairs.csv': ['mri_date', 'EXAMDATE_now'],
        'data_followup_6m_18m_1m.csv': ['mri_date', 'EXAMDATE']
    }
    
    for col in date_columns[os.path.basename(file1_path)]:
        df1[col] = pd.to_datetime(df1[col])
    for col in date_columns[os.path.basename(file2_path)]:
        df2[col] = pd.to_datetime(df2[col])
    
    # Create comparison keys
    df1['key'] = df1.apply(
        lambda row: f"{row['PTID']}_{row['mri_date'].strftime('%Y-%m-%d')}_{row['image_id']}_{row['EXAMDATE_now'].strftime('%Y-%m-%d')}", 
        axis=1
    )
    df2['key'] = df2.apply(
        lambda row: f"{row['PTID']}_{row['mri_date'].strftime('%Y-%m-%d')}_{row['image_id']}_{row['EXAMDATE'].strftime('%Y-%m-%d')}", 
        axis=1
    )
    
    # Print key statistics
    print("\nKey statistics:")
    print(f"Unique keys in {os.path.basename(file1_path)}:", df1['key'].nunique())
    print(f"Unique keys in {os.path.basename(file2_path)}:", df2['key'].nunique())
    
    # Find rows that are in df1 but not in df2
    only_in_df1 = df1[~df1['key'].isin(df2['key'])]
    
    # Find rows that are in df2 but not in df1
    only_in_df2 = df2[~df2['key'].isin(df1['key'])]
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total rows in {os.path.basename(file1_path)}: {len(df1)}")
    print(f"Total rows in {os.path.basename(file2_path)}: {len(df2)}")
    print(f"\nRows only in {os.path.basename(file1_path)}: {len(only_in_df1)}")
    print(f"Rows only in {os.path.basename(file2_path)}: {len(only_in_df2)}")
    
    # Save results to CSV files
    output_dir = "data/comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save full data for rows that don't match
    only_in_df1.to_csv(os.path.join(output_dir, "only_in_test_pairs.csv"), index=False)
    only_in_df2.to_csv(os.path.join(output_dir, "only_in_followup.csv"), index=False)
    
    # Save summary files with only key columns
    key_columns_df1 = ['PTID', 'mri_date', 'image_id', 'EXAMDATE_now']
    key_columns_df2 = ['PTID', 'mri_date', 'image_id', 'EXAMDATE']
    
    only_in_df1[key_columns_df1].to_csv(
        os.path.join(output_dir, "only_in_test_pairs_summary.csv"), 
        index=False
    )
    only_in_df2[key_columns_df2].to_csv(
        os.path.join(output_dir, "only_in_followup_summary.csv"), 
        index=False
    )
    
    print(f"\nResults saved to {output_dir}/")
    print("- Full data files:")
    print("  * only_in_test_pairs.csv")
    print("  * only_in_followup.csv")
    print("- Summary files (key columns only):")
    print("  * only_in_test_pairs_summary.csv")
    print("  * only_in_followup_summary.csv")
    
    # Print sample of differences
    if len(only_in_df1) > 0:
        print("\nSample of rows only in test_pairs.csv:")
        print(only_in_df1[key_columns_df1].head())
        print("\nSample keys from test_pairs.csv:")
        print(only_in_df1['key'].head())
    
    if len(only_in_df2) > 0:
        print("\nSample of rows only in data_followup_6m_18m_1m.csv:")
        print(only_in_df2[key_columns_df2].head())
        print("\nSample keys from data_followup_6m_18m_1m.csv:")
        print(only_in_df2['key'].head())

if __name__ == "__main__":
    file1_path = "data/test_pairs.csv"
    file2_path = "data/data_followup_6m_18m_1m.csv"
    compare_data_files(file1_path, file2_path) 