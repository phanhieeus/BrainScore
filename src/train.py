import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from data.dataset import BrainScoreDataModule
from models.fusion import FusionRegressor
import shutil  # Add this import for directory removal


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
    train_data_path: str,
    val_data_path: str,
    test_data_path: str,
    mri_dir: str,
    batch_size: int = 16,
    num_workers: int = 4,
    max_epochs: int = 100,
    accelerator: str = 'gpu' if torch.cuda.is_available() else 'cpu',
    devices: int = 1,
    precision: str = '16-mixed' if torch.cuda.is_available() else '32',
    log_dir: str = 'logs',
    checkpoint_dir: str = 'checkpoints',
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
    
    # Initialize model
    model = FusionRegressor(
        MRI_encoder_freeze=False,  # Allow MRI encoder to be trained
        clinical_dim=6,  # 3 demographic features + 3 current scores
        mri_dim=256,  # Output dimension of MRI encoder
        hidden_dims=[256, 128, 64, 128, 256],  # Hidden dimensions for fusion network
        dropout_rate=0.2  # Dropout rate for regularization
    )
    
    # Callbacks
    callbacks = [
        # Save best checkpoint based on validation loss
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='brainscore-{epoch:02d}-{val_loss:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            save_last=True
        ),
        # Early stopping if validation loss doesn't improve for 20 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=20,
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
        gradient_clip_val=1.0,  # Gradient clipping to prevent exploding gradients
        accumulate_grad_batches=2,  # Gradient accumulation to increase effective batch size
        log_every_n_steps=10,  # Log every 10 steps instead of default 50
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
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum number of epochs for training (default: 100)')
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'train_data_path': 'data/train_data.csv',
        'val_data_path': 'data/val_data.csv',
        'test_data_path': 'data/test_data.csv',
        'mri_dir': 'data/T1_biascorr_brain_data',
        'batch_size': 16,
        'num_workers': 4,
        'max_epochs': args.max_epochs,
        'log_dir': 'logs',
        'checkpoint_dir': 'checkpoints',
        'fast_dev_run': args.fast_dev_run
    }
    
    # Train model
    model, trainer = train_model(**config) 