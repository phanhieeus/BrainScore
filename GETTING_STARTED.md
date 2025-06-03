# BrainScore Project Guide

## 1. Environment Setup

1. Create virtual environment:
```bash
python -m venv venv
```

2. Activate virtual environment:
- On Windows:
```bash
venv\Scripts\activate
```
- On Linux/Mac:
```bash
source venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## 2. Data Requirements

Before starting, ensure you have the following data in the `data` directory:

1. MRI Images:
   - Directory: `data/T1_biascorr_brain_data/`
   - Contains subdirectories named `I{mri_id}/` (e.g., I165413, I285900, etc.)
   - Each subdirectory contains a `T1_biascorr_brain.nii.gz` file
   - All MRI images have been preprocessed using FSL's fsl_anat pipeline:
     * Skull stripping (removal of non-brain tissue)
     * Bias field correction
     * Registration to standard space
     * Normalization
     * Ready for direct use in the model

2. Processed Data Files:
   Each dataset (train/val/test) is split into two time points:
   
   a) 6-12 months data:
   - `data/train_6_12.csv`: Training set with normalized features
   - `data/val_6_12.csv`: Validation set with normalized features
   - `data/test_6_12.csv`: Test set with normalized features
   
   b) 6-18 months data:
   - `data/train_6_18.csv`: Training set with normalized features
   - `data/val_6_18.csv`: Validation set with normalized features
   - `data/test_6_18.csv`: Test set with normalized features
   
   Each CSV file contains the following normalized columns:
   * Patient info: PTID, mri_date, image_id
   * Test dates: EXAMDATE_now, EXAMDATE_future
   * Clinical features:
     - PTGENDER: Gender (0 for female, 1 for male)
     - age: Age at MRI time (normalized)
     - PTEDUCAT: Years of education (normalized)
     - ADAS11_now: Current ADAS11 score (normalized)
     - ADAS13_now: Current ADAS13 score (normalized)
     - MMSCORE_now: Current MMSE score (normalized)
     - DIAGNOSIS_now: Current diagnosis
   * Time data: time_lapsed (normalized)
   * Target scores (normalized):
     - ADAS11_future
     - ADAS13_future
     - MMSCORE_future

### Downloading Data

To download all required data files, run the download script:
```bash
./download_data.sh
```

This script will:
- Create the `data` directory if it doesn't exist
- Download the data zip file from Google Drive
- Extract all files to the correct locations
- Clean up the zip file after extraction

## 3. Project Structure

```
BrainScoreProject/
├── data/                      # Data directory
│   ├── T1_biascorr_brain_data/  # MRI images directory
│   │   ├── I13407/              # Directory for patient with mri_id = 13407
│   │   │   └── T1_biascorr_brain.nii.gz
│   │   └── ...
│   ├── train_6_12.csv           # Training set (6-12 months) with normalized features
│   ├── val_6_12.csv             # Validation set (6-12 months) with normalized features
│   ├── test_6_12.csv            # Test set (6-12 months) with normalized features
│   ├── train_6_18.csv           # Training set (6-18 months) with normalized features
│   ├── val_6_18.csv             # Validation set (6-18 months) with normalized features
│   └── test_6_18.csv            # Test set (6-18 months) with normalized features
│
├── src/                      # Source code
│   ├── data/                 # Data processing
│   │   ├── dataset.py             # Dataset class for model
│   │   └── denormalize_predictions.py # Convert predictions to original ranges
│   │
│   ├── models/              # Model definitions
│   │   ├── fusion.py       # Main fusion model
│   │   ├── encoders.py     # Encoder models
│   │   └── interactions.py # Interaction models
│   │
│   ├── train.py           # Training script
│   ├── predict.py         # Prediction script
│   └── analyze_errors.py  # Error analysis script
│
├── predictions/          # Model predictions
│   ├── test_predictions.csv           # Raw predictions
│   └── test_predictions_denormalized.csv # Denormalized predictions
├── analysis/            # Analysis results
│   ├── worst_10_*.csv   # Worst predictions analysis
│   └── worst_10_*.png   # Visualization plots
├── checkpoints/         # Model checkpoints
│   ├── brainscore-{epoch}-{val_loss}.ckpt # Best models
│   └── brainscore-final.ckpt # Final model
├── logs/               # Training logs
│   └── brainscore/    # TensorBoard logs
├── venv/              # Virtual environment
├── requirements.txt   # Python dependencies
├── download_data.sh  # Data download script
└── GETTING_STARTED.md # This guide
```

## 4. Data Preparation

### 4.1. Dataset for Model (dataset.py)

This file defines how to load and process data for the model:
```bash
python src/data/dataset.py
```

Main functions:
- Define `BrainScoreDataset` class to load data from CSV files
- Load and process MRI images with same transforms for all datasets (train, val, test):

  * Transforms:
    - LoadImaged: Read .nii.gz files
    - EnsureChannelFirstd: Ensure channel dimension is first
    - EnsureTyped: Ensure data type is float32
    - CenterSpatialCropd: Center crop 96x96x96 region
    - NormalizeIntensityd: Normalize image intensity (nonzero, channel-wise)

- Define `BrainScoreDataModule` class to manage data according to PyTorch Lightning standards:
  * Main parameters:
    - train_data_path: Path to train_data.csv
    - val_data_path: Path to val_data.csv
    - test_data_path: Path to test_data.csv
    - mri_dir: Directory containing MRI images
    - batch_size: Batch size for DataLoader (default: 16)
    - num_workers: Number of workers for DataLoader (default: 4)
  
  * Main methods:
    - setup(): Initialize datasets for training, validation and testing
    - train_dataloader(): Create DataLoader for training (with shuffle)
    - val_dataloader(): Create DataLoader for validation (without shuffle)
    - test_dataloader(): Create DataLoader for testing (without shuffle)
    - get_feature_dim(): Get dimension of clinical feature vector

  * Data processing:
    - Clinical data (8 values):
      * PTGENDER: Gender (0 for female, 1 for male)
      * age: Age at MRI time (normalized)
      * PTEDUCAT: Years of education (normalized)
      * ADAS11_now: Current ADAS11 score (normalized)
      * ADAS13_now: Current ADAS13 score (normalized)
      * MMSCORE_now: Current MMSE score (normalized)
      * DIAGNOSIS_now: Current diagnosis
      * time_lapsed: Time between tests (normalized)
    - Target data (3 values):
      * Future scores: ADAS11_future, ADAS13_future, MMSCORE_future
    - MRI data:
      * 3D image with size 96x96x96
      * Normalized intensity values
      * Channel-first format

## 5. Model Training

### 5.1. Training Process (train.py)

Run the training script:
```bash
# Run normal training
python src/train.py

# Run quick test
python src/train.py --fast-dev-run

# Run with custom number of epochs
python src/train.py --max-epochs 50
```

Main functions:
- Define `train_model` function with parameters:
  * train_data_path: Path to train_data.csv
  * val_data_path: Path to val_data.csv
  * test_data_path: Path to test_data.csv
  * mri_dir: Directory containing MRI images
  * batch_size: Batch size (default: 16)
  * num_workers: Number of workers (default: 4)
  * max_epochs: Maximum number of epochs (default: 100)
  * accelerator: Type of accelerator ('gpu' or 'cpu')
  * devices: Number of devices
  * precision: Precision ('16-mixed' or '32')
  * log_dir: Directory for logs
  * checkpoint_dir: Directory for checkpoints
  * fast_dev_run: Quick test mode

### 5.2. Training Configuration

- Model Architecture:
  * FusionRegressor with ResNet50 backbone
  * MRI Encoder: Extracts features from 3D MRI images
  * Clinical Encoder: Processes demographic data and current scores (6 features)
  * Time Encoder: Handles time differences between tests
  * Single-stage prediction:
    - Future scores: Predict ADAS11_future, ADAS13_future, MMSCORE_future

- Training Settings:
  * Optimizer: AdamW with learning rate 1e-4
  * Scheduler: ReduceLROnPlateau (reduce lr when loss plateaus)
  * Loss: MSE Loss for each future score
  * Metrics: 
    - MSE (Mean Squared Error)
    - MAE (Mean Absolute Error)
    - R² Score
  * Gradient clipping: 1.0
  * Gradient accumulation: 2 batches
  * Mixed precision training: 16-bit (FP16)
  * Logging frequency: Every 10 steps
  * Device: GPU if available, otherwise CPU

- Callbacks:
  * ModelCheckpoint: Save 3 best models and final model
  * EarlyStopping: Stop training if validation loss doesn't improve for 20 epochs

- Logging:
  * TensorBoard logger
  * Log every 10 steps
  * Save logs to logs directory
  * Track metrics for future predictions

### 5.3. Quick Test Mode

The `--fast-dev-run` option is useful for:
- Quick code testing
- Debugging
- Checking data loading
- Verifying model architecture

When using this mode:
- Run only a few batches
- Don't save checkpoints
- Don't save logs
- Run only 1 epoch
- Skip test set evaluation

### 5.4. Training Output

After training, you'll find:

1. Logs Directory (`logs/`):
   * Structure: `logs/brainscore/version_X/`
   * New version created for each run
   * Contains metrics for future predictions:
     - Loss: Total loss for future predictions
     - MSE: For each score (ADAS11, ADAS13, MMSE)
     - MAE: For each score
     - R² Score: For each score
     - Learning rate
   * Usage:
     ```bash
     tensorboard --logdir logs
     ```
   * Access: http://localhost:6006
   * Used to monitor training progress and compare runs

2. Checkpoints Directory (`checkpoints/`):
   * `brainscore-{epoch}-{val_loss}.ckpt`: Best models
   * `brainscore-final.ckpt`: Final model
   * `last.ckpt`: Model at last epoch
   * Usage:
     ```python
     # Load model from checkpoint
     model = FusionRegressor.load_from_checkpoint(
         checkpoint_path='checkpoints/brainscore-epoch=04-val_loss=265.0553.ckpt'
     )
     ```
   * Used to save best models, continue training, deploy models

3. Test Results:
   * Printed after training completion
   * Shows metrics for future predictions:
     - MSE, MAE, R² for ADAS11, ADAS13, MMSE
     - Overall performance summary

## 6. Model Prediction and Analysis

### 6.1. Generate Predictions (predict.py)

Run the prediction script:
```bash
# Generate predictions for test set
python src/predict.py --dataset test

# Generate predictions for validation set
python src/predict.py --dataset val

# Generate predictions for training set
python src/predict.py --dataset train
```

Main functions:
- Automatically find best model checkpoint based on validation loss
- Load model and generate predictions for specified dataset
- Save predictions to `predictions/{dataset_type}_predictions.csv` with columns:
  * Patient info: image_id, mri_date, PTGENDER, age, PTEDUCAT
  * Test dates: EXAMDATE_now, EXAMDATE_future
  * Time difference: time_lapsed
  * Future scores: ADAS11_future_pred, ADAS13_future_pred, MMSCORE_future_pred
  * Ground truth: ADAS11_future_true, ADAS13_future_true, MMSCORE_future_true
  * Errors: ADAS11_future_error, ADAS13_future_error, MMSCORE_future_error

### 6.2. Denormalize Predictions (denormalize_predictions.py)

Run the denormalization script:
```bash
# Denormalize test set predictions
python src/data/denormalize_predictions.py --dataset test

# Denormalize validation set predictions
python src/data/denormalize_predictions.py --dataset val

# Denormalize training set predictions
python src/data/denormalize_predictions.py --dataset train
```

Main functions:
- Read normalized predictions from `predictions/{dataset_type}_predictions.csv`
- Convert predictions back to original ranges:
  * ADAS11: 0-70
  * ADAS13: 0-85
  * MMSE: 0-30
- Save denormalized predictions to `predictions/{dataset_type}_predictions_denormalized.csv`
- Print detailed statistics for each score:
  * Mean prediction and ground truth
  * Mean absolute error
  * Root mean square error
  * R² score
  * Min/max values for predictions and ground truth
  * Error statistics (range and standard deviation)

### 6.3. Visualize Predictions (visualize_predictions.py)

Run the visualization script:
```bash
# Visualize test set predictions
python src/visualize_predictions.py --dataset test

# Visualize validation set predictions
python src/visualize_predictions.py --dataset val

# Visualize training set predictions
python src/visualize_predictions.py --dataset train
```

Main functions:
- Load predictions from `predictions/{dataset_type}_predictions.csv`
- Create two types of visualizations:
  1. Predictions vs Ground Truth:
     * Scatter plots for future scores (ADAS11, ADAS13, MMSE)
     * Perfect prediction line for reference
     * R² score for each plot
     * Saved as `visualizations/{dataset_type}_predictions_vs_ground_truth.png`
  
  2. Error Distributions:
     * Histograms of prediction errors for each future score
     * Mean and standard deviation of errors
     * Saved as `visualizations/{dataset_type}_error_distributions.png`

### 6.4. Analyze Worst Predictions (analyze_errors.py)

Run the analysis script:
```bash
# Analyze test set worst predictions
python src/analyze_errors.py --dataset test --n 10

# Analyze validation set worst predictions
python src/analyze_errors.py --dataset val --n 10

# Analyze training set worst predictions
python src/analyze_errors.py --dataset train --n 10
```

Main functions:
- Read predictions from `predictions/{dataset_type}_predictions.csv`
- For each future score type (ADAS11, ADAS13, MMSE):
  * Find n worst predictions based on MAE
  * Save detailed analysis to CSV files in `analysis/` directory
  * Create visualizations:
    - Scatter plot of predictions vs ground truth
    - Error distribution histogram
  * Print detailed information about each worst prediction:
    - Patient demographics
    - Time between visits
    - Prediction and ground truth values
    - Error magnitude

Output files in `analysis/` directory:
1. CSV files:
   * `{dataset_type}_worst_{n}_ADAS11_future_predictions.csv`
   * `{dataset_type}_worst_{n}_ADAS13_future_predictions.csv`
   * `{dataset_type}_worst_{n}_MMSCORE_future_predictions.csv`

2. Visualization files:
   * `{dataset_type}_worst_{n}_ADAS11_future_predictions.png`
   * `{dataset_type}_worst_{n}_ADAS13_future_predictions.png`
   * `{dataset_type}_worst_{n}_MMSCORE_future_predictions.png`

Each visualization includes:
- Scatter plot comparing predictions vs ground truth
- Error distribution histogram
- Perfect prediction line for reference
- Clear labels and titles
- Mean and standard deviation of errors

## 7. Common Issues and Solutions

1. File not found errors:
   - Check paths to data files
   - Ensure scripts are run in correct order
   - Check if mri_ids in data have corresponding image files in T1_biascorr_brain_data directory

2. MRI image loading errors:
   - Check image file format (.nii.gz)
   - Check path to image directory
   - Ensure subdirectory names in T1_biascorr_brain_data follow format "I{mri_id}"

3. Training errors:
   - Check GPU memory if using GPU
   - Reduce batch size if out of memory
   - Check appropriate number of workers for CPU
   - Use fast_dev_run for quick debugging

4. Model loading errors:
   - Check PyTorch and PyTorch Lightning versions
   - Check if model structure matches
   - Check path to checkpoint