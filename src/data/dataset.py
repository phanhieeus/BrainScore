import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import nibabel as nib
from monai.transforms import (
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandSpatialCropd,
    RandFlipd,
    RandRotate90d,
    NormalizeIntensityd,
    CenterSpatialCropd,
    Compose
)
from datetime import datetime


class BrainScoreDataset(Dataset):
    def __init__(self, mri_dir, data_path, transforms=None):
        """
        Dataset for BrainScore
        
        Args:
            mri_dir (str): Directory containing MRI images
            data_path (str): Path to train_6_12.csv, val_6_12.csv, test_6_12.csv or train_6_18.csv, val_6_18.csv, test_6_18.csv
            transforms (str): "Train" for training transforms, None for validation/test transforms
        """
        super().__init__()
        self.clinical_columns = [
            'PTGENDER', 'age', 'PTEDUCAT',
            'ADAS11_now', 'ADAS13_now', 'MMSCORE_now',
            'DIAGNOSIS_now', 'time_lapsed'
        ]
        self.target_columns = [
            'ADAS11_future', 'ADAS13_future', 
            'MMSCORE_future'
        ]
        self.clinical_data = pd.read_csv(data_path)
        self.mri_dir = mri_dir
        
        # Define transforms
        if transforms == "Train":
            self.transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"]),
                RandSpatialCropd(keys=["image"], roi_size=[64, 64, 64], random_size=False),
                RandFlipd(keys=["image"], spatial_axis=0, prob=0.3),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.3),
                RandFlipd(keys=["image"], spatial_axis=2, prob=0.3),
                RandRotate90d(keys=["image"], max_k=3, prob=0.3),
                NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ])
        else:
            self.transforms = Compose([
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                EnsureTyped(keys=["image"]),
                CenterSpatialCropd(keys=["image"], roi_size=[64, 64, 64]),
                NormalizeIntensityd(keys="image", nonzero=False, channel_wise=True)
            ])
        
        # Filter out samples without image files
        valid_indices = []
        for idx, row in self.clinical_data.iterrows():
            image_path = os.path.join(self.mri_dir, f"I{row['image_id']}", "T1_biascorr_brain.nii.gz")
            if os.path.exists(image_path):
                valid_indices.append(idx)
            else:
                print(f"Warning: Image file not found for mri_id {row['image_id']}")
        
        # Keep only samples with image files
        self.clinical_data = self.clinical_data.iloc[valid_indices].reset_index(drop=True)
    
    def __len__(self):
        return len(self.clinical_data)
    
    def __getitem__(self, idx):
        # Get data from row
        row = self.clinical_data.iloc[idx]
        
        # Get clinical data
        clinical_data = row[self.clinical_columns]
        clinical_data = clinical_data.to_numpy()
        clinical_data = torch.from_numpy(clinical_data.astype(np.float32))
        
        # Get targets
        targets = row[self.target_columns]
        targets = targets.to_numpy()
        targets = torch.from_numpy(targets.astype(np.float32))
        
        # Load and process MRI image
        image_id = row['image_id']
        image_path = os.path.join(self.mri_dir, f"I{image_id}", "T1_biascorr_brain.nii.gz")
        
        # Create dictionary for MONAI transforms
        data_dict = {"image": image_path}
        transformed = self.transforms(data_dict)
        img = transformed["image"]
        
        return img, clinical_data, targets


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
            train_data_path (str): Path to train_6_12.csv or train_6_18.csv
            val_data_path (str): Path to val_6_12.csv or val_6_18.csv
            test_data_path (str): Path to test_6_12.csv or test_6_18.csv
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
                mri_dir=self.mri_dir,
                data_path=self.train_data_path,
                transforms="Train"
            )
            
            # Initialize validation dataset
            self.val_dataset = BrainScoreDataset(
                mri_dir=self.mri_dir,
                data_path=self.val_data_path,
                transforms=None
            )
            
            # Save clinical columns from train dataset
            self.clinical_columns = self.train_dataset.clinical_columns
            
        if stage == 'test' or stage is None:
            # Initialize test dataset
            self.test_dataset = BrainScoreDataset(
                mri_dir=self.mri_dir,
                data_path=self.test_data_path,
                transforms=None
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
    # Test DataModule with only validation data
    data_module = BrainScoreDataModule(
        train_data_path="data/val_6_18.csv",  # Using val data temporarily
        val_data_path="data/val_6_18.csv",
        test_data_path="data/val_6_18.csv",  # Using val data temporarily
        mri_dir="data/T1_biascorr_brain_data"
    )
    
    # Setup datasets
    data_module.setup()
    
    # Test validation dataloader
    val_loader = data_module.val_dataloader()
    
    # Print validation dataset size
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    
    # Test a batch
    batch = next(iter(val_loader))
    mri, clinical, targets = batch
    
    print("\nBatch shapes:")
    print(f"MRI: {mri.shape}")
    print(f"Clinical: {clinical.shape}")
    print(f"Targets: {targets.shape}")
    
