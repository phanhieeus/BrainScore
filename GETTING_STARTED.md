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

2. Cognitive Test Scores:
   - File: `data/c1_c2_cognitive_score.csv`
   - Columns:
     * PTID: Patient ID
     * VISCODE: Visit code
     * VISCODE2: Secondary visit code
     * EXAMDATE: Test date
     * DIAGNOSIS: Patient diagnosis
     * image_id: MRI image ID
     * mri_date: MRI scan date
     * ADAS11: ADAS-Cog 11 score
     * ADAS13: ADAS-Cog 13 score
     * MMSCORE: Mini-Mental State Examination score
     * CDGLOBAL: Clinical Dementia Rating global score

3. Demographics Data:
   - File: `data/c1_c2_demographics.csv`
   - Columns:
     * PTID: Patient ID
     * PTGENDER: Gender (1 for male, 0 for female)
     * PTDOBYY: Year of birth
     * PTEDUCAT: Years of education

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
│   ├── c1_c2_cognitive_score.csv # Cognitive test scores
│   ├── c1_c2_demographics.csv    # Demographics data
│   ├── test_pairs.csv           # Processed test pairs
│   ├── test_pairs_normalized.csv # Normalized test pairs
│   ├── train_data.csv           # Training set
│   ├── val_data.csv            # Validation set
│   └── test_data.csv           # Test set
│
├── src/                      # Source code
│   ├── data/                 # Data processing
│   │   ├── create_test_pairs.py    # Create dataset with test pairs
│   │   ├── normalize_test_pairs.py # Normalize test pairs data
│   │   ├── split_data.py          # Split data into train/val/test sets
│   │   ├── dataset.py             # Dataset class for model
│   │   └── denormalize_predictions.py # Convert predictions to original ranges
│   │
│   ├── models/              # Model definitions
│   │   ├── fusion.py       # Main fusion model
│   │   ├── encoders.py     # Encoder models
│   │   └── interactions.py # Interaction models
│   │
│   ├── dataprocessing/     # Data analysis
│   │   └── analyze_score_changes.py # Analyze cognitive score changes
│   │
│   ├── train.py           # Training script
│   ├── predict.py         # Prediction script
│   └── analyze_errors.py  # Error analysis script
│
├── notebooks/             # Jupyter notebooks for analysis
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

### 4.1. Create Dataset with Test Pairs (create_test_pairs.py)

This is the first file to run for data preparation:
```bash
python src/data/create_test_pairs.py
```

Main functions:
- Read data from 2 files: `c1_c2_cognitive_score.csv` (test scores) and `c1_c2_demographics.csv` (patient information)
- Check and filter data:
  * Remove rows with negative cognitive test scores (ADAS11, ADAS13, MMSCORE)
  * Iterate through all mri_ids in the data
  * Check if `T1_biascorr_brain.nii.gz` exists in corresponding directory
  * Keep only mri_ids with corresponding image files
- Split data into two parts:
  * MRI data (PTID, mri_date, image_id)
  * Test data (PTID, EXAMDATE, ADAS11, ADAS13, MMSCORE)
- For each patient and MRI:
  * Find tests within 30 days of MRI date (both before and after)
  * Find tests between 180-540 days after MRI date
  * Create pairs of tests (one near, one future)
  * Combine with demographics data (gender, age, education)
- Calculate additional features:
  * Age at MRI time
  * Time elapsed between tests
  * Gender (0 for female, 1 for male)
- Save results to `test_pairs.csv` with columns:
  * Patient info: PTID, mri_date, image_id
  * Test dates: EXAMDATE_now, EXAMDATE_future
  * Current scores: ADAS11_now, ADAS13_now, MMSCORE_now
  * Future scores: ADAS11_future, ADAS13_future, MMSCORE_future
  * Demographics: PTGENDER, age, PTEDUCAT
  * Time difference: time_lapsed
- Print detailed statistics about:
  * Total data points and number of patients
  * Gender distribution
  * Age distribution
  * Education distribution
  * Time between tests

### 4.2. Normalize Test Pairs Data (normalize_test_pairs.py)

Run script to normalize the test pairs data:
```bash
python src/data/normalize_test_pairs.py
```

Main functions:
- Read data from `test_pairs.csv`
- Normalize clinical and cognitive scores using min-max scaling:
  * Age: 50-100 years
  * Education years (PTEDUCAT): 5-25 years
  * ADAS11 scores (now and future): 0-70
  * ADAS13 scores (now and future): 0-85
  * MMSE scores (now and future): 0-30
- Formula used for normalization:
  ```
  normalized_value = (value - min_val) / (max_val - min_val)
  ```
- Keep other columns unchanged:
  * Patient info: PTID, mri_date, image_id
  * Test dates: EXAMDATE_now, EXAMDATE_future
  * Demographics: PTGENDER
  * Time difference: time_lapsed
- Save normalized data to `test_pairs_normalized.csv`
- Save score ranges to `score_ranges.json` for later use in denormalization

Denormalization:
- After model prediction, convert normalized predictions back to original ranges
- Use `denormalize_predictions()` function:
  ```python
  # Load normalized predictions
  normalized_predictions = pd.read_csv('predictions.csv')
  
  # Convert back to original ranges
  denormalized_predictions = denormalize_predictions(normalized_predictions)
  
  # Save results
  denormalized_predictions.to_csv('denormalized_predictions.csv', index=False)
  ```
- Formula used for denormalization:
  ```
  original_value = normalized_value * (max_val - min_val) + min_val
  ```
- Automatically handles all columns that were previously normalized
- Preserves original column names and structure
- Creates a new DataFrame without modifying the original predictions

### 4.3. Split Data into Train, Validation and Test Sets (split_data.py)

Run script to split data:
```bash
python src/data/split_data.py
```

Main functions:
- Read normalized data from `test_pairs_normalized.csv`
- Split data into 3 sets: train (80%), validation (10%), and test (10%)
- Ensure:
  * No patients shared between sets
  * Maintain similar demographics distribution between validation and test sets
  * Representative distribution of:
    - Gender ratio
    - Age (normalized)
    - Education level (normalized)
    - Time between tests
- Save results to 3 files: `train_data.csv`, `val_data.csv`, and `test_data.csv`
- Print detailed statistics about the split, including:
  * Number of patients and data points in each set
  * Demographics analysis (gender, age, education)
  * Time between tests statistics
  * Distribution differences between validation and test sets

### 4.4. Dataset for Model (dataset.py)

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
    - Clinical data (3 values):
      * PTGENDER: Gender (0 for female, 1 for male)
      * age: Age at MRI time (normalized)
      * PTEDUCAT: Years of education (normalized)
    - Time data (1 value):
      * time_lapsed: Time between tests (in days)
    - Target data (6 values):
      * Current scores: ADAS11_now, ADAS13_now, MMSCORE_now
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
  * Clinical Encoder: Processes demographic data (gender, age, education)
  * Time Encoder: Handles time differences between tests
  * Two-stage prediction:
    - Current scores: Predict ADAS11_now, ADAS13_now, MMSCORE_now
    - Future scores: Predict ADAS11_future, ADAS13_future, MMSCORE_future

- Training Settings:
  * Optimizer: AdamW with learning rate 1e-4
  * Scheduler: ReduceLROnPlateau (reduce lr when loss plateaus)
  * Loss: MSE Loss for each score
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
  * EarlyStopping: Stop training if validation loss doesn't improve for 10 epochs

- Logging:
  * TensorBoard logger
  * Log every 50 steps
  * Save logs to logs directory
  * Track metrics for both current and future predictions

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
   * Contains metrics for both current and future predictions:
     - Loss: Total loss combining current and future predictions
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
   * Shows metrics for both current and future predictions:
     - MSE, MAE, R² for ADAS11, ADAS13, MMSE
     - Separate metrics for current and future scores
     - Overall performance summary

## 6. Model Prediction and Analysis

### 6.1. Generate Predictions (predict.py)

Run the prediction script:
```bash
python src/predict.py
```

Main functions:
- Automatically find best model checkpoint based on validation loss
- Load model and generate predictions for test set
- Save predictions to `predictions/test_predictions.csv` with columns:
  * Patient info: image_id, mri_date, PTGENDER, age, PTEDUCAT
  * Test dates: EXAMDATE_now, EXAMDATE_future
  * Time difference: time_lapsed
  * Current scores: ADAS11_now_pred, ADAS13_now_pred, MMSCORE_now_pred
  * Future scores: ADAS11_future_pred, ADAS13_future_pred, MMSCORE_future_pred
  * Ground truth: ADAS11_now_true, ADAS13_now_true, MMSCORE_now_true, etc.
  * Errors: ADAS11_now_error, ADAS13_now_error, MMSCORE_now_error, etc.

### 6.2. Denormalize Predictions (denormalize_predictions.py)

Run the denormalization script:
```bash
python src/data/denormalize_predictions.py
```

Main functions:
- Read normalized predictions from `test_predictions.csv`
- Convert predictions back to original ranges:
  * ADAS11: 0-70
  * ADAS13: 0-85
  * MMSE: 0-30
- Save denormalized predictions to `test_predictions_denormalized.csv`
- Print detailed statistics for each score:
  * Mean prediction and ground truth
  * Mean absolute error
  * Root mean square error
  * R² score
  * Min/max values for predictions and ground truth
  * Error statistics

### 6.3. Analyze Worst Predictions (analyze_errors.py)

Run the analysis script:
```bash
python src/analyze_errors.py
```

Main functions:
- Read denormalized predictions
- For each score type (ADAS11, ADAS13, MMSE for both current and future):
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
   * `worst_10_ADAS11_now_predictions.csv`
   * `worst_10_ADAS13_now_predictions.csv`
   * `worst_10_MMSCORE_now_predictions.csv`
   * `worst_10_ADAS11_future_predictions.csv`
   * `worst_10_ADAS13_future_predictions.csv`
   * `worst_10_MMSCORE_future_predictions.csv`

2. Visualization files:
   * `worst_10_ADAS11_now_predictions.png`
   * `worst_10_ADAS13_now_predictions.png`
   * `worst_10_MMSCORE_now_predictions.png`
   * `worst_10_ADAS11_future_predictions.png`
   * `worst_10_ADAS13_future_predictions.png`
   * `worst_10_MMSCORE_future_predictions.png`

Each visualization includes:
- Scatter plot comparing predictions vs ground truth
- Error distribution histogram
- Perfect prediction line for reference
- Clear labels and titles

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