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
    CenterSpatialCropd,
    NormalizeIntensityd,
    Compose
)
from datetime import datetime


class BrainScoreDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        mri_dir, 
        is_train=True
    ):
        """
        Dataset for BrainScore
        
        Args:
            data_path (str): Path to train_data.csv, val_data.csv or test_data.csv
            mri_dir (str): Directory containing MRI images
            is_train (bool): True for training set, False for validation/test set
        """
        # Load data
        self.data = pd.read_csv(data_path, sep=',')
        
        # Convert date columns
        self.data['mri_date'] = pd.to_datetime(self.data['mri_date'])
        self.data['EXAMDATE_now'] = pd.to_datetime(self.data['EXAMDATE_now'])
        self.data['EXAMDATE_future'] = pd.to_datetime(self.data['EXAMDATE_future'])
        
        # Select columns for clinical data (including current scores)
        self.clinical_columns = [
            'PTGENDER', 'age', 'PTEDUCAT',
            'ADAS11_now', 'ADAS13_now', 'MMSCORE_now'  # Added current scores
        ]
        
        # Convert clinical data to float32
        for col in self.clinical_columns:
            self.data[col] = self.data[col].astype(np.float32)
        
        # Convert time_lapsed to float32
        self.data['time_lapsed'] = self.data['time_lapsed'].astype(np.float32)
        
        # Convert target columns to float32
        # We predict only future scores
        self.target_columns = [
            'ADAS11_future', 'ADAS13_future', 'MMSCORE_future'
        ]
            
        for col in self.target_columns:
            self.data[col] = self.data[col].astype(np.float32)
        
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
        
        # Use same transform for all sets (train, val, test)
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
        
        # Get normalized clinical data (now includes current scores)
        clinical = torch.from_numpy(
            row[self.clinical_columns].values.astype(np.float32)
        )
        
        # Get time (only 2 dimensions: batch_size and time value)
        time_lapsed = torch.tensor(
            [row['time_lapsed']],
            dtype=torch.float32
        )
        
        # Get targets (only future scores)
        targets = torch.from_numpy(
            row[self.target_columns].values.astype(np.float32)
        )
        
        return mri, clinical, time_lapsed, targets


class BrainScoreDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_data_path: str,
        val_data_path: str,
        test_data_path: str,
        mri_dir: str,
        batch_size: int = 16,
        num_workers: int = 4
    ):
        """
        DataModule for BrainScore
        
        Args:
            train_data_path (str): Path to train_data.csv
            val_data_path (str): Path to val_data.csv
            test_data_path (str): Path to test_data.csv
            mri_dir (str): Directory containing MRI images
            batch_size (int): Batch size for DataLoader (default: 16)
            num_workers (int): Number of workers for DataLoader
        """
        super().__init__()
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.mri_dir = mri_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def setup(self, stage=None):
        """Initialize datasets for training, validation and testing"""
        if stage == 'fit' or stage is None:
            # Initialize train dataset
            self.train_dataset = BrainScoreDataset(
                data_path=self.train_data_path,
                mri_dir=self.mri_dir,
                is_train=True
            )
            
            # Initialize validation dataset
            self.val_dataset = BrainScoreDataset(
                data_path=self.val_data_path,
                mri_dir=self.mri_dir,
                is_train=False
            )
            
            # Save clinical columns from train dataset
            self.clinical_columns = self.train_dataset.clinical_columns
            
        if stage == 'test' or stage is None:
            # Initialize test dataset
            self.test_dataset = BrainScoreDataset(
                data_path=self.test_data_path,
                mri_dir=self.mri_dir,
                is_train=False
            )
    
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
        """DataLoader for validation"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def test_dataloader(self):
        """DataLoader for testing"""
        return DataLoader(
            self.test_dataset,
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
        val_data_path="data/val_data.csv",
        test_data_path="data/test_data.csv",
        mri_dir="data/T1_biascorr_brain_data"
    )
    
    # Setup datasets
    data_module.setup()
    
    # Test dataloaders
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    
    # Print dataset sizes
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    
    # Test a batch
    batch = next(iter(train_loader))
    mri, clinical, time, targets = batch
    
    print("\nBatch shapes:")
    print(f"MRI: {mri.shape}")
    print(f"Clinical: {clinical.shape}")
    print(f"Time: {time.shape}")
    print(f"Targets: {targets.shape}")
    
