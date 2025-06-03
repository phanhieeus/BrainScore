import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data.dataset import BrainScoreDataModule
from models.fusion import FusionRegressor
import shutil  # Add this import for directory removal

# Constants for training configuration
DEFAULT_BATCH_SIZE = 4
DEFAULT_NUM_WORKERS = 4
DEFAULT_MAX_EPOCHS = 100
DEFAULT_ACCUMULATE_GRAD_BATCHES = 4
DEFAULT_GRADIENT_CLIP_VAL = 1.0
DEFAULT_LOG_EVERY_N_STEPS = 10
DEFAULT_EARLY_STOPPING_PATIENCE = 20
DEFAULT_SAVE_TOP_K = 3

# Constants for paths
DEFAULT_LOG_DIR = 'logs'
DEFAULT_CHECKPOINT_DIR = 'checkpoints'
DEFAULT_TRAIN_DATA_PATH = 'data/train_6_18.csv'
DEFAULT_VAL_DATA_PATH = 'data/val_6_18.csv'
DEFAULT_TEST_DATA_PATH = 'data/test_6_18.csv'
DEFAULT_MRI_DIR = 'data/T1_biascorr_brain_data'


def clean_directories(log_dir: str, checkpoint_dir: str):
    """
    Clean up old logs and checkpoints
    
    Args:
        log_dir (str): Directory for logs
        checkpoint_dir (str): Directory for checkpoints
    """
    # Remove old logs
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
        print(f"Removed old logs from {log_dir}")
    
    # Remove old checkpoints
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
        print(f"Removed old checkpoints from {checkpoint_dir}")
    
    # Create fresh directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    print("Created fresh directories for logs and checkpoints")


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
    log_dir: str = DEFAULT_LOG_DIR,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    fast_dev_run: bool = False
):
    """
    Train BrainScore model to predict future cognitive scores
    
    Args:
        train_data_path (str): Path to train_data.csv
        val_data_path (str): Path to val_data.csv
        test_data_path (str): Path to test_data.csv
        mri_dir (str): Directory containing MRI images
        batch_size (int): Batch size
        num_workers (int): Number of workers for DataLoader
        max_epochs (int): Maximum number of epochs
        accelerator (str): Type of accelerator ('gpu' or 'cpu')
        devices (int): Number of devices
        precision (str): Precision ('16-mixed' or '32')
        log_dir (str): Directory for logs
        checkpoint_dir (str): Directory for checkpoints
        fast_dev_run (bool): Quick test mode
    """
    # Clean up old logs and checkpoints
    if not fast_dev_run:  # Only clean if not in fast dev run mode
        clean_directories(log_dir, checkpoint_dir)
    
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
    clinical_dim = data_module.get_feature_dim()  # Now returns 8 features
    
    # Initialize model
    model = FusionRegressor(
        mri_dim=512,  # SwinUNETR output dimension
        clinical_dim=clinical_dim,  # 8 features from dataset
        hidden_dims=[256, 512, 256, 128],  # Hidden dimensions for fusion network
        dropout_rate=0.1  # Updated default dropout rate
    )
    
    # Callbacks
    callbacks = [
        # Save best checkpoint based on validation loss
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='brainscore-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=DEFAULT_SAVE_TOP_K,
            save_last=True
        ),
        # Early stopping if validation loss doesn't improve for 20 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=DEFAULT_EARLY_STOPPING_PATIENCE,
            mode='min',
            verbose=True
        )
    ]
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=log_dir,
        name='brainscore',
        version=None  # Auto-increment version
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=DEFAULT_GRADIENT_CLIP_VAL,  # Gradient clipping to prevent exploding gradients
        accumulate_grad_batches=DEFAULT_ACCUMULATE_GRAD_BATCHES,  # Gradient accumulation to increase effective batch size
        log_every_n_steps=DEFAULT_LOG_EVERY_N_STEPS,  # Log every 10 steps instead of default 50
        fast_dev_run=fast_dev_run  # Quick test mode
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Test model
    if not fast_dev_run:
        print("\nTesting model on test set...")
        trainer.test(model, data_module.test_dataloader())
    
    # Save final model
    trainer.save_checkpoint(os.path.join(checkpoint_dir, 'brainscore-final.ckpt'))
    
    # Print model and training information
    print("\nTraining completed!")
    
    # Only print best model info if not fast_dev_run
    if not fast_dev_run:
        print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    else:
        print("Fast dev run completed - no model saved")
    
    return model, trainer


if __name__ == "__main__":
    import argparse
    
    # Create parser for command line arguments
    parser = argparse.ArgumentParser(description='Train BrainScore model to predict future cognitive scores')
    parser.add_argument('--fast-dev-run', action='store_true', help='Run a quick test of the training code')
    parser.add_argument('--max-epochs', type=int, default=DEFAULT_MAX_EPOCHS, help='Maximum number of epochs for training')
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'train_data_path': DEFAULT_TRAIN_DATA_PATH,
        'val_data_path': DEFAULT_VAL_DATA_PATH,
        'test_data_path': DEFAULT_TEST_DATA_PATH,
        'mri_dir': DEFAULT_MRI_DIR,
        'batch_size': DEFAULT_BATCH_SIZE,
        'num_workers': DEFAULT_NUM_WORKERS,
        'max_epochs': args.max_epochs,
        'log_dir': DEFAULT_LOG_DIR,
        'checkpoint_dir': DEFAULT_CHECKPOINT_DIR,
        'fast_dev_run': args.fast_dev_run
    }
    
    # Train model
    model, trainer = train_model(**config) 