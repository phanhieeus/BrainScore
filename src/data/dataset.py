import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImage,
    ScaleIntensity,
    Resize,
    RandRotate,
    RandZoom,
    RandAdjustContrast,
    ToTensor,
)
from typing import Dict, List, Tuple, Optional

class BrainScoreDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        cognitive_scores: pd.DataFrame,
        demographics: pd.DataFrame,
        transform: Optional[Compose] = None,
        is_train: bool = True
    ):
        """
        Custom dataset for brain MRI images and cognitive scores.
        
        Args:
            image_dir: Directory containing MRI images
            cognitive_scores: DataFrame with cognitive scores
            demographics: DataFrame with demographic information
            transform: Optional MONAI transforms
            is_train: Whether this is training data (for augmentation)
        """
        self.image_dir = image_dir
        self.cognitive_scores = cognitive_scores
        self.demographics = demographics
        self.is_train = is_train
        
        # Merge cognitive scores with demographics
        self.data = pd.merge(
            cognitive_scores,
            demographics,
            on="PTID",
            how="inner"
        )
        
        # Calculate age at MRI
        self.data["age"] = pd.to_datetime(self.data["mri_date"]).dt.year - self.data["PTDOBYY"]
        
        # Standardize demographic features
        self.demographic_features = ["PTGENDER", "age", "PTEDUCAT"]
        self.cognitive_features = ["ADAS11", "ADAS13", "MMSCORE", "CDGLOBAL"]
        
        # Calculate mean and std for standardization
        self.demographic_means = self.data[self.demographic_features].mean()
        self.demographic_stds = self.data[self.demographic_features].std()
        
        # Setup transforms
        if transform is None:
            self.transform = self._get_default_transforms()
        else:
            self.transform = transform
            
    def _get_default_transforms(self) -> Compose:
        """Get default transforms for the dataset."""
        transforms = [
            LoadImage(image_only=True),
            ScaleIntensity(),
            Resize(spatial_size=(96, 96, 96)),
            ToTensor()
        ]
        
        if self.is_train:
            transforms.extend([
                RandRotate(range_x=10, range_y=10, range_z=10, prob=0.5),
                RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
                RandAdjustContrast(prob=0.5)
            ])
            
        return Compose(transforms)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample from the dataset."""
        row = self.data.iloc[idx]
        
        # Load and transform MRI image
        image_path = os.path.join(
            self.image_dir,
            f"I{row['image']}",
            "T1_biascorr_brain.nii.gz"
        )
        image = self.transform(image_path)
        
        # Get demographic features
        demographic_data = row[self.demographic_features].values
        demographic_data = (demographic_data - self.demographic_means) / self.demographic_stds
        demographic_tensor = torch.FloatTensor(demographic_data)
        
        # Get cognitive scores
        cognitive_scores = row[self.cognitive_features].values
        cognitive_tensor = torch.FloatTensor(cognitive_scores)
        
        return {
            "image": image,
            "demographic": demographic_tensor,
            "cognitive_scores": cognitive_tensor,
            "ptid": row["PTID"],
            "mri_date": row["mri_date"]
        } 