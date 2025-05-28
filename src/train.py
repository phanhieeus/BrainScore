import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import BrainScoreDataModule
from models.fusion import FusionRegressor


def train_model(
    train_data_path: str,
    val_data_path: str,
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
    Train BrainScore model
    
    Args:
        train_data_path (str): Path to train_data.csv
        val_data_path (str): Path to test_data.csv
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
    # Create logs and checkpoints directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize DataModule
    data_module = BrainScoreDataModule(
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        mri_dir=mri_dir,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Setup datasets
    data_module.setup()
    
    # Initialize model
    model = FusionRegressor(
        mri_dim=2048,  # ResNet50 feature dimension
        clinical_dim=data_module.get_feature_dim(),
        time_dim=64,
        hidden_dims=[512, 256],
        output_dim=4  # ADAS11, ADAS13, MMSCORE, CDGLOBAL
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
        # Early stopping if validation loss doesn't improve for 10 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=10,
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
        accelerator=accelerator,
        devices=devices,
        max_epochs=max_epochs,
        callbacks=callbacks,
        logger=logger,
        precision=precision,
        gradient_clip_val=1.0,  # Gradient clipping to prevent exploding gradients
        accumulate_grad_batches=2,  # Gradient accumulation to increase effective batch size
        log_every_n_steps=50,  # Log every 50 steps
        fast_dev_run=fast_dev_run  # Quick test mode
    )
    
    # Train model
    trainer.fit(model, data_module)
    
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
    parser = argparse.ArgumentParser(description='Train BrainScore model')
    parser.add_argument('--fast-dev-run', action='store_true', help='Run a quick test of the training code')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum number of epochs for training (default: 100)')
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'train_data_path': 'data/train_data.csv',
        'val_data_path': 'data/test_data.csv',
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
    
    # Print model and training information
    print("\nTraining completed!")
    
    # Only print best model info if not fast_dev_run
    if not args.fast_dev_run:
        print(f"Best model saved at: {trainer.checkpoint_callback.best_model_path}")
        print(f"Best validation loss: {trainer.checkpoint_callback.best_model_score:.4f}")
    else:
        print("Fast dev run completed - no model saved") 