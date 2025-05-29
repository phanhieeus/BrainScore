import os
import torch
import pandas as pd
import numpy as np
from pytorch_lightning import Trainer
from data.dataset import BrainScoreDataModule
from models.fusion import FusionRegressor
import logging
from sklearn.metrics import r2_score
import argparse

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

def predict_dataset(model, dataloader, device):
    """Generate predictions for a dataset"""
    try:
        current_predictions = []
        future_predictions = []
        ground_truth = []
        
        # Set model to evaluation mode
        model = model.to(device)
        model.eval()
        
        # Generate predictions
        with torch.no_grad():
            for batch in dataloader:
                # Move data to device
                mri_data = batch[0].to(device)
                demographic_data = batch[1].to(device)
                time_lapsed = batch[2].to(device)
                targets = batch[3]
                
                # Get predictions
                current_scores, future_scores = model(mri_data, demographic_data, time_lapsed)
                
                # Convert predictions to numpy
                current_pred = torch.stack(current_scores).cpu().numpy().T
                future_pred = torch.stack(future_scores).cpu().numpy().T
                
                # Store predictions and ground truth
                current_predictions.append(current_pred)
                future_predictions.append(future_pred)
                ground_truth.append(targets.numpy())
        
        # Concatenate all batches
        current_predictions = np.concatenate(current_predictions, axis=0)
        future_predictions = np.concatenate(future_predictions, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)
        
        # Combine current and future predictions
        predictions = np.concatenate([current_predictions, future_predictions], axis=1)
        
        return predictions, ground_truth
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def create_prediction_file(data, predictions, ground_truth, output_path):
    """Create prediction file with original and predicted scores"""
    try:
        # Create a copy of data with only essential columns
        essential_columns = [
            'image_id', 'mri_date', 'EXAMDATE_now', 'EXAMDATE_future',
            'PTGENDER', 'age', 'PTEDUCAT', 'time_lapsed'
        ]
        result_df = data[essential_columns].copy()
        
        # Add prediction columns
        score_columns = [
            'ADAS11_now', 'ADAS13_now', 'MMSCORE_now',
            'ADAS11_future', 'ADAS13_future', 'MMSCORE_future'
        ]
        
        # Add predictions and ground truth
        for i, col in enumerate(score_columns):
            result_df[f'{col}_pred'] = predictions[:, i]
            result_df[f'{col}_true'] = ground_truth[:, i]
            result_df[f'{col}_error'] = predictions[:, i] - ground_truth[:, i]
            
            # Log metrics
            mae = np.mean(np.abs(result_df[f'{col}_error']))
            mse = np.mean(result_df[f'{col}_error'] ** 2)
            r2 = r2_score(result_df[f'{col}_true'], result_df[f'{col}_pred'])
            logger.info(f"{col} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
        
        # Reorder columns for better readability
        column_order = []
        for col in score_columns:
            column_order.extend([f'{col}_pred', f'{col}_true', f'{col}_error'])
        
        # Final column order: essential columns first, then predictions
        final_columns = essential_columns + column_order
        result_df = result_df[final_columns]
        
        # Save results
        result_df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error creating prediction file: {str(e)}")
        raise

def find_best_checkpoint(checkpoint_dir):
    """Find the best model checkpoint based on validation loss"""
    try:
        # Get all checkpoint files
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
        
        # Filter only training checkpoints
        training_checkpoints = [f for f in checkpoint_files if 'val_loss=' in f]
        if not training_checkpoints:
            raise FileNotFoundError(f"No training checkpoint files found in {checkpoint_dir}")
        
        # Sort by validation loss
        training_checkpoints.sort(key=lambda x: float(x.split('val_loss=')[-1].split('.')[0]))
        best_checkpoint = os.path.join(checkpoint_dir, training_checkpoints[0])
        
        logger.info(f"Found {len(training_checkpoints)} checkpoints")
        logger.info(f"Best checkpoint: {best_checkpoint}")
        logger.info(f"Validation loss: {training_checkpoints[0].split('val_loss=')[-1].split('.')[0]}")
        
        return best_checkpoint
        
    except Exception as e:
        logger.error(f"Error finding best checkpoint: {str(e)}")
        raise

def main():
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Generate predictions for BrainScore model')
        parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test'], required=True,
                          help='Dataset to generate predictions for')
        args = parser.parse_args()
        
        # Configuration
        config = {
            'train_data_path': 'data/train_data.csv',
            'val_data_path': 'data/val_data.csv',
            'test_data_path': 'data/test_data.csv',
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
            test_data_path=config['test_data_path'],
            mri_dir=config['mri_dir'],
            batch_size=config['batch_size'],
            num_workers=config['num_workers']
        )
        
        # Setup datasets
        data_module.setup()
        
        # Load data based on dataset type
        if args.dataset == 'train':
            data = pd.read_csv(config['train_data_path'])
            dataloader = data_module.train_dataloader()
        elif args.dataset == 'val':
            data = pd.read_csv(config['val_data_path'])
            dataloader = data_module.val_dataloader()
        else:  # test
            data = pd.read_csv(config['test_data_path'])
            dataloader = data_module.test_dataloader()
        
        # Find best model checkpoint
        best_checkpoint = find_best_checkpoint(config['checkpoint_dir'])
        
        # Load best model
        model = load_best_model(best_checkpoint)
        
        # Generate predictions
        predictions, ground_truth = predict_dataset(model, dataloader, device)
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Create prediction file
        output_path = os.path.join(config['output_dir'], f'{args.dataset}_predictions.csv')
        create_prediction_file(data, predictions, ground_truth, output_path)
        
        logger.info(f"Prediction completed successfully for {args.dataset} set")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise

if __name__ == '__main__':
    main() 