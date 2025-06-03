import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from data.dataset import BrainScoreDataModule
from models.fusion import FusionRegressor

class DeviceDebugCallback(Callback):
    def __init__(self):
        self.device = None
    
    def on_train_start(self, trainer, pl_module):
        self.device = trainer.strategy.root_device
        print(f"\nTraining device: {self.device}")
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        mri, clinical, targets = batch
        print(f"\nBatch {batch_idx} devices:")
        print(f"MRI: {mri.device}")
        print(f"Clinical: {clinical.device}")
        print(f"Targets: {targets.device}")
        
        # Check model devices
        print("\nModel devices:")
        print(f"MRI Encoder: {next(pl_module.mri_encoder.parameters()).device}")
        print(f"Clinical Encoder: {next(pl_module.clinical_encoder.parameters()).device}")
        print(f"Fusion Network: {next(pl_module.fusion_network.parameters()).device}")
        print(f"Future Heads: {next(pl_module.future_heads.parameters()).device}")
        
        # Check intermediate tensors
        with torch.no_grad():
            mri_features = pl_module.mri_encoder(mri)
            clinical_features = pl_module.clinical_encoder(clinical)
            print("\nFeature devices:")
            print(f"MRI features: {mri_features.device}")
            print(f"Clinical features: {clinical_features.device}")
            
            combined = torch.cat([mri_features, clinical_features], dim=1)
            print(f"Combined features: {combined.device}")
            
            shared = pl_module.fusion_network(combined)
            print(f"Shared features: {shared.device}")
            
            predictions = [head(shared) for head in pl_module.future_heads]
            print(f"Predictions: {[p.device for p in predictions]}")

def debug_training(
    train_data_path: str = 'data/train_6_18.csv',
    val_data_path: str = 'data/val_6_18.csv',
    test_data_path: str = 'data/test_6_18.csv',
    mri_dir: str = 'data/T1_biascorr_brain_data',
    batch_size: int = 2,  # Reduced batch size for debugging
    num_workers: int = 4,
    max_epochs: int = 1,  # Just one epoch for debugging
    accelerator: str = 'gpu' if torch.cuda.is_available() else 'cpu',
    devices: int = 1,
    precision: str = '16-mixed' if torch.cuda.is_available() else '32'
):
    """
    Debug training process to check device placement of tensors
    """
    # Initialize DataModule
    data_module = BrainScoreDataModule(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        test_data_path=test_data_path,
        mri_dir=mri_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Setup datasets
    data_module.setup()
    
    # Get clinical feature dimension from dataset
    clinical_dim = data_module.get_feature_dim()
    
    # Initialize model
    model = FusionRegressor(
        mri_dim=512,
        clinical_dim=clinical_dim,
        hidden_dims=[256, 512, 256, 128],
        dropout_rate=0.1
    )
    
    # Create debug callback
    debug_callback = DeviceDebugCallback()
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=[debug_callback],
        fast_dev_run=True  # Just run one batch
    )
    
    # Train model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    debug_training() 