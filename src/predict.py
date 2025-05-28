import os
import torch
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from data.dataset import BrainScoreDataModule
from models.fusion import FusionRegressor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_best_model(checkpoint_path):
    """Load the best model from checkpoint"""
    try:
        model = FusionRegressor.load_from_checkpoint(checkpoint_path)
        model.eval()
        logger.info(f"Successfully loaded model from {checkpoint_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def predict_validation(model, data_module, device):
    """Generate predictions for validation set"""
    try:
        predictions = []
        ground_truth = []
        
        # Get validation dataloader
        val_dataloader = data_module.val_dataloader()
        
        # Set model to evaluation mode
        model = model.to(device)
        model.eval()
        
        # Generate predictions
        with torch.no_grad():
            for batch in val_dataloader:
                # Move data to device
                mri_data = batch[0].to(device)  # First element is MRI data
                demographic_data = batch[1].to(device)  # Second element is demographic data
                time_lapsed = batch[2].to(device)  # Third element is time_lapsed
                targets = batch[3]  # Fourth element is targets
                
                # Get predictions
                outputs = model(mri_data, demographic_data, time_lapsed)
                
                # Store predictions and ground truth
                predictions.append(outputs.cpu().numpy())
                ground_truth.append(targets.numpy())
        
        # Concatenate all batches
        predictions = np.concatenate(predictions, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)
        
        return predictions, ground_truth
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def create_prediction_file(val_data, predictions, ground_truth, output_path):
    """Create prediction file with original and predicted scores"""
    try:
        # Create a copy of validation data
        result_df = val_data.copy()
        
        # Add prediction columns
        score_columns = ['ADAS11', 'ADAS13', 'MMSCORE', 'CDGLOBAL']
        for i, col in enumerate(score_columns):
            result_df[f'{col}_predict'] = predictions[:, i]
            result_df[f'{col}_ground_truth'] = ground_truth[:, i]
            
            # Calculate error
            result_df[f'{col}_error'] = result_df[f'{col}_predict'] - result_df[f'{col}_ground_truth']
            
            # Log metrics
            mae = np.mean(np.abs(result_df[f'{col}_error']))
            mse = np.mean(result_df[f'{col}_error'] ** 2)
            logger.info(f"{col} - MAE: {mae:.4f}, MSE: {mse:.4f}")
        
        # Save results
        result_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating prediction file: {str(e)}")
        raise

def main():
    try:
        # Configuration
        config = {
            'train_data_path': 'data/train_data.csv',
            'val_data_path': 'data/test_data.csv',
            'mri_dir': 'data/T1_biascorr_brain_data',
            'batch_size': 16,
            'num_workers': 4,
            'checkpoint_dir': 'checkpoints',
            'output_dir': 'predictions'
        }
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Initialize DataModule
        data_module = BrainScoreDataModule(
            train_data_path=config['train_data_path'],
            val_data_path=config['val_data_path'],
            mri_dir=config['mri_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # Setup datasets
        data_module.setup()
        
        # Load validation data
        val_data = pd.read_csv(config['val_data_path'])
        
        # Find best checkpoint
        checkpoint_dir = config['checkpoint_dir']
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
            
        # Filter only training checkpoints (format: brainscore-epoch=XX-val_loss=XXX.XXXX.ckpt)
        training_checkpoints = [f for f in checkpoint_files if 'val_loss=' in f]
        if not training_checkpoints:
            raise FileNotFoundError(f"No training checkpoint files found in {checkpoint_dir}")
            
        # Sort by validation loss
        training_checkpoints.sort(key=lambda x: float(x.split('val_loss=')[-1].split('.')[0]))
        best_checkpoint = os.path.join(checkpoint_dir, training_checkpoints[0])
        logger.info(f"Selected checkpoint: {best_checkpoint}")
        
        # Load best model
        model = load_best_model(best_checkpoint)
        
        # Generate predictions
        predictions, ground_truth = predict_validation(model, data_module, device)
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Create prediction file
        output_path = os.path.join(config['output_dir'], 'validation_predictions.csv')
        create_prediction_file(val_data, predictions, ground_truth, output_path)
        
        logger.info("Prediction completed successfully")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 