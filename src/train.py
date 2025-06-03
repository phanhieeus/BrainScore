import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from data.dataset import BrainScoreDataModule
from models.fusion import FusionRegressor
import shutil

# Constants for training configuration
DEFAULT_BATCH_SIZE = 2
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_EPOCHS = 100
DEFAULT_ACCUMULATE_GRAD_BATCHES = 8
DEFAULT_GRADIENT_CLIP_VAL = 1.0
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_SAVE_TOP_K = 3

# Constants for paths
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_TRAIN_DATA_PATH = 'data/train_6_18.csv'
DEFAULT_VAL_DATA_PATH = 'data/val_6_18.csv'
DEFAULT_TEST_DATA_PATH = 'data/test_6_18.csv'
DEFAULT_MRI_DIR = 'data/T1_biascorr_brain_data'


def clean_directories(checkpoint_dir: str):
    """Clean up old checkpoints"""
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Removed old checkpoints from {checkpoint_dir}")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Created fresh directory for checkpoints")


def train_model(
    train_data_path: str = DEFAULT_TRAIN_DATA_PATH,
    val_data_path: str = DEFAULT_VAL_DATA_PATH,
    test_data_path: str = DEFAULT_TEST_DATA_PATH,
    mri_dir: str = DEFAULT_MRI_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    accelerator: str = 'gpu' if torch.cuda.is_available() else 'cpu',
    devices: int = 1,
    precision: str = '16-mixed' if torch.cuda.is_available() else '32',
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    fast_dev_run: bool = False
):
    # Clean up old checkpoints
    if not fast_dev_run:
        clean_directories(checkpoint_dir)
    
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
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='brainscore-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=DEFAULT_SAVE_TOP_K,
            save_last=True
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=DEFAULT_EARLY_STOPPING_PATIENCE,
            mode='min',
            verbose=True
        )
    ]
    
    # Trainer with memory optimizations
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        gradient_clip_val=DEFAULT_GRADIENT_CLIP_VAL,
        accumulate_grad_batches=DEFAULT_ACCUMULATE_GRAD_BATCHES,
        fast_dev_run=fast_dev_run,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=True,
        detect_anomaly=False,
        benchmark=False
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    if not fast_dev_run:
        print("\nTesting model on test set...")
        trainer.test(model, data_module.test_dataloader())
    
    # Save final model
    trainer.save_checkpoint(os.path.join(checkpoint_dir, 'brainscore-final.ckpt'))
    
    print("\nTraining completed!")
    
    if not fast_dev_run:
        print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    else:
        print("Fast dev run completed - no model saved")
    
    return model, trainer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BrainScore model to predict future cognitive scores')
    parser.add_argument('--fast-dev-run', action='store_true', help='Run a quick test of the training code')
    parser.add_argument('--max-epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='Maximum number of epochs for training')
    args = parser.parse_args()
    
    config = {
        'train_data_path': DEFAULT_TRAIN_DATA_PATH,
        'val_data_path': DEFAULT_VAL_DATA_PATH,
        'test_data_path': DEFAULT_TEST_DATA_PATH,
        'mri_dir': DEFAULT_MRI_DIR,
        'batch_size': DEFAULT_BATCH_SIZE,
        'num_workers': DEFAULT_NUM_WORKERS,
        'max_epochs': args.max_epochs,
        'checkpoint_dir': DEFAULT_CHECKPOINT_DIR,
        'fast_dev_run': args.fast_dev_run
    }
    
    model, trainer = train_model(**config) 