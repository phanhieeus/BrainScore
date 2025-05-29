import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shutil

import warnings
warnings.filterwarnings('ignore')


# Define the path to the data directory
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

def load_data(filename):
    file_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        return df
    else:
        raise FileNotFoundError(f"The file {file_path} does not exist.")

def save_data(df, filename):
    file_path = os.path.join(DATA_DIR, filename)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")
        
def demographic_processing(df):
    df = df.drop_duplicates(subset=['PTID'], keep='first')
    print(f"Finished processing demographic data, {len(df)} records remaining.")
    return df
    
def cognitive_processing(df):
    # Colect unique image IDs from the dataframe
    img_id = df['image_id'].values
    img_id = np.unique(img_id)
    
    # Get image folders that match the image IDs
    image_id_folder = []
    all_folders = os.listdir(os.path.join(DATA_DIR, 'T1_biascorr_brain_data'))
    for folder in all_folders:
        folder_id = folder[1:]
        folder_id = int(folder_id)
        if folder_id in img_id:
            image_id_folder.append(folder_id)
    
    image_id_folder = np.unique(image_id_folder)
    # Get dataframe if image_id is in the folder
    df = df[df['image_id'].isin(image_id_folder)]
    # Drop the data invalid
    df = df[df['ADAS11'] >= 0]
    df = df[df['ADAS13'] >= 0]
    df = df[df['MMSCORE'] >= 0]
    df = df[df['CDGLOBAL'] >= 0]
    print(f"Finished processing cognitive data, {len(df)} records remaining.")
    return df
    
def merge_data(demographic_df, cognitive_df):
    merged_df = pd.merge(cognitive_df, demographic_df, on='PTID', how='inner')
    merged_df['EXAMDATE'] = pd.to_datetime(merged_df['EXAMDATE'])
    merged_df['AGE'] = merged_df['EXAMDATE'].dt.year - merged_df['PTDOBYY']
    merged_df = merged_df.dropna()
    print(f"Finished merging data, {len(merged_df)} records remaining.")
    return merged_df

def get_future_data(group):
    from_month = 6
    to_month = 18
    
    followup_adas11 = []
    followup_adas13 = []
    followup_mmscore = []
    followup_cdglobal = []
    time_followup = []

    for i, row in group.iterrows():
        exam_date = row['EXAMDATE']
        future_rows = group[(group['EXAMDATE'] > exam_date) & 
                            (group['EXAMDATE'] >= exam_date + pd.Timedelta(days=from_month*30)) &
                            (group['EXAMDATE'] <= exam_date + pd.Timedelta(days=to_month*30))]

        if not future_rows.empty:
            next_row = future_rows.iloc[0]
            followup_adas11.append(next_row['ADAS11'])
            followup_adas13.append(next_row['ADAS13'])
            followup_mmscore.append(next_row['MMSCORE'])
            followup_cdglobal.append(next_row['CDGLOBAL'])
            time_followup.append(future_rows['EXAMDATE'].iloc[0] - exam_date)
        else:
            followup_adas11.append(np.nan)
            followup_adas13.append(np.nan)
            followup_mmscore.append(np.nan)
            followup_cdglobal.append(np.nan)
            time_followup.append(np.nan)

    group[f'ADAS11_{from_month}_{to_month}'] = followup_adas11
    group[f'ADAS13_{from_month}_{to_month}'] = followup_adas13
    group[f'MMSCORE_{from_month}_{to_month}'] = followup_mmscore
    group[f'CDGLOBAL_{from_month}_{to_month}'] = followup_cdglobal
    group[f'TIME_FOLLOWUP'] = time_followup
    return group

def normalize_scores(scores, score_ranges):
    normalized = {}
    for score_name, value in scores.items():
        min_val, max_val = score_ranges[score_name]
        normalized[score_name] = (value - min_val) / (max_val - min_val)
    return normalized

def create_data_csv():
    demographic_df = load_data('c1_c2_demographics.csv')
    cognitive_df = load_data('c1_c2_cognitive_score.csv')
    
    demographic_df = demographic_processing(demographic_df)
    cognitive_df = cognitive_processing(cognitive_df)
    
    merged_df = merge_data(demographic_df, cognitive_df)
    
    # Print debug info before filtering
    print("\nDebug: Data before filtering")
    print("Total rows:", len(merged_df))
    print("\nSample rows for patient 002_S_2073:")
    print(merged_df[merged_df['PTID'] == '002_S_2073'][['PTID', 'EXAMDATE', 'mri_date', 'image_id']].sort_values('EXAMDATE'))
    
    df = merged_df.groupby('PTID').apply(get_future_data).reset_index(drop=True)
    
    df['TIME_FOLLOWUP'] = df['TIME_FOLLOWUP'].dt.days
    # Normalize scores
    score_range = {
        'ADAS11': (0, 70), # ADAS11 score range
        'ADAS13': (0, 85), # ADAS13 score range
        'MMSCORE': (0, 30), # MMSE score range
        'CDGLOBAL': (0, 3), # CDR Global score range
        'ADAS11_6_18': (0, 70), # ADAS11 score range for follow-up
        'ADAS13_6_18': (0, 85), # ADAS13 score range for follow-up
        'MMSCORE_6_18': (0, 30), # MMSE score range for follow-up
        'CDGLOBAL_6_18': (0, 3), # CDR Global score range for follow-up
        'AGE': (50, 100), # Age range
        'PTEDUCAT': (5, 25), # Years of education range
        'TIME_FOLLOWUP': (0, 30) # days
    }
    # Normalize scores
    for name, value in score_range.items():
        df[name] = df[name].apply(lambda x: normalize_scores({name: x}, score_range)[name])

    df['mri_date'] = pd.to_datetime(df['mri_date'])
    df['EXAMDATE'] = pd.to_datetime(df['EXAMDATE'])
    
    # Print debug info before time filtering
    print("\nDebug: Data before time filtering")
    print("Total rows:", len(df))
    print("\nSample rows for patient 002_S_2073:")
    print(df[df['PTID'] == '002_S_2073'][['PTID', 'EXAMDATE', 'mri_date', 'image_id']].sort_values('EXAMDATE'))
    
    month = 1
    df = df[df['mri_date'] - df['EXAMDATE'] >= pd.Timedelta(days=-month*30)]
    df = df[df['mri_date'] - df['EXAMDATE'] <= pd.Timedelta(days=month*30)]
    df = df.dropna()
    
    # Print debug info after time filtering
    print("\nDebug: Data after time filtering")
    print("Total rows:", len(df))
    print("\nSample rows for patient 002_S_2073:")
    print(df[df['PTID'] == '002_S_2073'][['PTID', 'EXAMDATE', 'mri_date', 'image_id']].sort_values('EXAMDATE'))
    
    # Save the processed data
    save_data(df, 'data_followup_6m_18m_1m.csv')
    print("Data processing complete. Processed data saved as 'data_followup_6m_18m_1m.csv'.")

if __name__ == "__main__":
    create_data_csv()
