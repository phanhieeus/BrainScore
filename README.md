# Brain Score Project 🧠

A deep learning project for predicting future cognitive test scores from brain MRI images and demographics data.
Link to report: https://docs.google.com/document/d/14LXy6imsjAijm7Upmz41WT-_iGXutofiIkkS5eesSwI/edit?usp=sharing

## 📋 Overview

This project aims to predict four cognitive test scores (ADAS11, ADAS13, MMSCORE, CDGLOBAL) using:
- 3D brain MRI images
- Clinical data (gender, age, education)
- Time difference between MRI scan and cognitive test

The model combines these different types of data using a fusion architecture with:
- ResNet50 backbone for MRI feature extraction
- Clinical data encoder
- Time elapsed encoder
- Interaction layers to capture relationships between features

## 📊 Results Summary

| Model Architecture | Data Processing | MSE (train/val/test) | MAE (train/val/test) | R² Score (train/val/test) |
|-------------------|-----------------|---------------------|---------------------|--------------------------|
| | | | | |

## 🚀 Quick Start

For detailed setup and usage instructions, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## 📁 Project Structure

```
BrainScoreProject/
├── data/                      # Data directory
│   ├── T1_biascorr_brain_data/  # MRI images
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
│   │   ├── create_test_pairs.py
│   │   ├── normalize_test_pairs.py
│   │   ├── split_data.py
│   │   ├── dataset.py
│   │   └── denormalize_predictions.py
│   │
│   ├── models/              # Model definitions
│   │   ├── fusion.py       # Main fusion model
│   │   ├── encoders.py     # Encoder models
│   │   └── interactions.py # Interaction models
│   │
│   ├── dataprocessing/     # Data analysis
│   │   └── analyze_score_changes.py
│   │
│   ├── train.py           # Training script
│   ├── predict.py         # Prediction script
│   └── analyze_errors.py  # Error analysis script
│
├── notebooks/             # Jupyter notebooks
├── predictions/          # Model predictions
├── analysis/            # Analysis results
├── checkpoints/         # Model checkpoints
├── logs/               # Training logs
├── venv/              # Virtual environment
├── requirements.txt   # Python dependencies
├── download_data.sh  # Data download script
└── GETTING_STARTED.md # Detailed guide
```

## 🧪 Cognitive Tests

The model predicts scores from four cognitive tests:

1. **ADAS11**: Alzheimer's Disease Assessment Scale - 11 items
   - Measures cognitive impairment
   - Higher scores indicate more severe impairment

2. **ADAS13**: Alzheimer's Disease Assessment Scale - 13 items
   - Extended version of ADAS11
   - Includes additional memory and language tasks

3. **MMSCORE**: Mini-Mental State Examination
   - Brief 30-point test
   - Assesses cognitive impairment
   - Lower scores indicate more severe impairment

4. **CDGLOBAL**: Clinical Dementia Rating Scale
   - Measures dementia severity
   - Higher scores indicate more severe dementia

## 👥 Contributors

| Student ID | Name |
|------------|------|
| | Phan Văn Hiếu (Phan Hieu) |
| | Nguyễn Đức Minh (Nguyen Duc Minh) |
| | Trần Tiến Nam (Tran Tien Nam) |
| | Vũ Đình Quang Huy (Vu Dinh Quang Huy) |
| | Nguyễn Trọng Khánh (Nguyen Trong Khanh) |
| | Bùi Văn Khải (Bui Van Khai) |

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{brain_score_project,
  author = {Phan Hieu},
  title = {Brain Score},
  year = {2025},
  url = {https://github.com/phanhieus/BrainScore}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
