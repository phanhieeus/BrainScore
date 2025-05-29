import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    Compose,
    CenterSpatialCropd
)
from datetime import datetime
import nibabel as nib
import SimpleITK as sitk


class BrainScoreDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        mri_dir, 
        is_train=True,
        age_min=50,
        age_max=100,
        educ_min=5,
        educ_max=25
    ):
        """
        Dataset for BrainScore
        
        Args:
            data_path (str): Path to train_data.csv or test_data.csv
            mri_dir (str): Directory containing MRI images
            is_train (bool): True for training set, False for validation set
            age_min (int): Minimum age for min-max scaling (default: 50)
            age_max (int): Maximum age for min-max scaling (default: 100)
            educ_min (int): Minimum years of education for min-max scaling (default: 5)
            educ_max (int): Maximum years of education for min-max scaling (default: 25)
        """
        # Load data
        self.data = pd.read_csv(data_path, sep=',')
        
        # Convert date columns
        self.data['mri_date'] = pd.to_datetime(self.data['mri_date'])
        self.data['EXAMDATE_now'] = pd.to_datetime(self.data['EXAMDATE_now'])
        self.data['EXAMDATE_future'] = pd.to_datetime(self.data['EXAMDATE_future'])
        
        # Select columns for clinical data
        self.clinical_columns = ['PTGENDER', 'age', 'PTEDUCAT']
        
        # Normalize clinical data
        # Gender: 1 for male, 0 for female (already normalized)
        # Age: min-max scaling with custom values
        # Education years: min-max scaling with custom values
        self.data['age'] = (self.data['age'] - age_min) / (age_max - age_min)
        self.data['PTEDUCAT'] = (self.data['PTEDUCAT'] - educ_min) / (educ_max - educ_min)
        
        # Convert clinical data to float32
        for col in self.clinical_columns:
            self.data[col] = self.data[col].astype(np.float32)
        
        # Convert target columns to float32
        target_columns = ['ADAS11', 'ADAS13', 'MMSCORE', 'CDGLOBAL']
        for col in target_columns:
            self.data[col] = self.data[col].astype(np.float32)
        
        # Convert test_mri_time_diff to float32
        self.data['test_mri_time_diff'] = self.data['test_mri_time_diff'].astype(np.float32)
        
        self.mri_dir = mri_dir
        self.is_train = is_train
        
        # Filter out samples without image files
        valid_indices = []
        for idx, row in self.data.iterrows():
            image_path = os.path.join(self.mri_dir, f"I{row['image_id']}", "T1_biascorr_brain.nii.gz")
            if os.path.exists(image_path):
                valid_indices.append(idx)
            else:
                print(f"Warning: Image file not found for mri_id {row['image_id']}")
        
        # Keep only samples with image files
        self.data = self.data.iloc[valid_indices].reset_index(drop=True)
        
        # Define transforms based on is_train
        if is_train:
            self.transform = Compose([
                LoadImaged(keys=["image"], image_only=True),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"], dtype=torch.float32),
                RandSpatialCropd(keys=["image"], roi_size=[96, 96, 96], random_size=False),
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.1),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.1),
                RandFlipd(keys=["image"], spatial_axis=2, prob=0.1),
                RandRotate90d(keys=["image"], max_k=3, prob=0.1),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ])
        else:
            # Transform for validation
            self.transform = Compose([
                LoadImaged(keys=["image"], image_only=True),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"], dtype=torch.float32),
                CenterSpatialCropd(keys=["image"], roi_size=[96, 96, 96]),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
            ])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get data from row
        row = self.data.iloc[idx]
        
        # Prepare data for transform
        data_dict = {
            "image": os.path.join(self.mri_dir, f"I{row['image_id']}", "T1_biascorr_brain.nii.gz")
        }
        
        # Apply transform
        transformed = self.transform(data_dict)
        mri = transformed["image"]
        
        # Get normalized clinical data
        clinical = torch.from_numpy(
            row[self.clinical_columns].values.astype(np.float32)
        )
        
        # Get time (only 2 dimensions: batch_size and time value)
        time_lapsed = torch.tensor(
            [row['test_mri_time_diff']],
            dtype=torch.float32
        )
        
        # Get targets (4 values to predict)
        targets = torch.from_numpy(
            row[['ADAS11', 'ADAS13', 'MMSCORE', 'CDGLOBAL']].values.astype(np.float32)
        )
        
        return mri, clinical, time_lapsed, targets


class BrainScoreDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        mri_dir: str,
        batch_size: int = 16,  # Changed default batch size to 16
        num_workers: int = 4,
        age_min: int = 50,
        age_max: int = 100,
        educ_min: int = 5,
        educ_max: int = 25
    ):
        """
        DataModule for BrainScore
        
        Args:
            train_data_path (str): Path to train_data.csv
            val_data_path (str): Path to test_data.csv (used as validation set)
            mri_dir (str): Directory containing MRI images
            batch_size (int): Batch size for DataLoader (default: 16)
            num_workers (int): Number of workers for DataLoader
            age_min (int): Minimum age for min-max scaling (default: 50)
            age_max (int): Maximum age for min-max scaling (default: 100)
            educ_min (int): Minimum years of education for min-max scaling (default: 5)
            educ_max (int): Maximum years of education for min-max scaling (default: 25)
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.mri_dir = mri_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.age_min = age_min
        self.age_max = age_max
        self.educ_min = educ_min
        self.educ_max = educ_max
        
    def setup(self, stage=None):
        """Initialize datasets for training and validation"""
        if stage == 'fit' or stage is None:
            # Initialize train dataset
            self.train_dataset = BrainScoreDataset(
                data_path=self.train_data_path,
                mri_dir=self.mri_dir,
                is_train=True,
                age_min=self.age_min,
                age_max=self.age_max,
                educ_min=self.educ_min,
                educ_max=self.educ_max
            )
            
            # Initialize validation dataset
            self.val_dataset = BrainScoreDataset(
                data_path=self.val_data_path,
                mri_dir=self.mri_dir,
                is_train=False,
                age_min=self.age_min,
                age_max=self.age_max,
                educ_min=self.educ_min,
                educ_max=self.educ_max
            )
            
            # Save clinical columns from train dataset
            self.clinical_columns = self.train_dataset.clinical_columns
    
    def train_dataloader(self):
        """DataLoader for training"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """DataLoader for validation (using test_data.csv)"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def get_feature_dim(self):
        """Return dimension of clinical feature vector"""
        return len(self.train_dataset.clinical_columns)


if __name__ == "__main__":
    # Test DataModule
    data_module = BrainScoreDataModule(
        train_data_path="data/train_data.csv",
        val_data_path="data/test_data.csv",
        mri_dir="data/T1_biascorr_brain_data"
    )
    
    # Setup datasets
    data_module.setup()
    
    # Test dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Print dataset sizes
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    mri, clinical, time, targets = batch
    
    print("\nBatch shapes:")
    print(f"MRI: {mri.shape}")
    print(f"Clinical: {clinical.shape}")
    print(f"Time: {time.shape}")
    print(f"Targets: {targets.shape}")
    
    # In statistics about time
    print(f"\nAverage: {data_module.train_dataset.data['test_mri_time_diff'].mean():.2f}")
