# Brain Score Project 🧠

A deep learning project for predicting future cognitive test scores from brain MRI images and clinical data.
Link to report: https://docs.google.com/document/d/14LXy6imsjAijm7Upmz41WT-_iGXutofiIkkS5eesSwI/edit?usp=sharing

## 📋 Overview

This project aims to predict three cognitive test scores (ADAS11, ADAS13, MMSCORE) using:
- 3D brain MRI images (T1-weighted)
- Clinical data (gender, age, education, current scores)
- Time difference between MRI scan and cognitive test

The model architecture consists of:
- MRI Encoder: 3D CNN for feature extraction from brain scans
- Clinical Encoder: MLP for processing demographic and current scores
- Fusion Regressor: Combines features from both encoders to predict future scores

## 🚀 Quick Start

For detailed setup and usage instructions, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## 📁 Project Structure

```
BrainScore/
├── data/                      # Data directory
│   ├── T1_biascorr_brain_data/  # MRI images
│   ├── train_6_12.csv          # Training data (6-12 months)
│   ├── val_6_12.csv           # Validation data (6-12 months)
│   ├── test_6_12.csv          # Test data (6-12 months)
│   ├── train_6_18.csv          # Training data (6-18 months)
│   ├── val_6_18.csv           # Validation data (6-18 months)
│   └── test_6_18.csv          # Test data (6-18 months)
│
├── src/                      # Source code
│   ├── data/                 # Data processing
│   │   ├── dataset.py        # PyTorch dataset implementation
│   │   └── denormalize_predictions.py # Prediction denormalization
│   │
│   ├── models/              # Model definitions
│   │   ├── fusion.py        # Fusion model architecture
│   │   ├── encoders.py      # MRI and clinical encoders
│   │   └── interactions.py  # Feature interaction layers
│   │
│   ├── analysis/           # Analysis tools
│   │   ├── analyze_errors.py # Error analysis and metrics
│   │   └── visualize_predictions.py # Prediction visualization
│   │
│   ├── train.py           # Training script
│   ├── predict.py         # Prediction script
│   └── debug_device.py    # Device debugging utilities
│
├── notebooks/             # Jupyter notebooks for analysis
├── checkpoints/         # Model checkpoints
├── logs/               # Training logs
├── requirements.txt   # Python dependencies
├── download_data.sh  # Data download script
└── GETTING_STARTED.md # Detailed guide
```

## 🧪 Problem Statement

The project addresses the challenge of predicting future cognitive test scores using multimodal data:

1. **Input Data**:
   - 3D T1-weighted MRI brain scans
   - Clinical features:
     - Demographics (gender, age, education)
     - Current cognitive scores (ADAS11, ADAS13, MMSCORE)
     - Time elapsed between scans

2. **Target Variables**:
   - Future ADAS11 score
   - Future ADAS13 score
   - Future MMSCORE

3. **Time Windows**:
   - Short-term prediction (6-12 months)
   - Long-term prediction (6-18 months)

4. **Model Architecture**:
   - MRI Encoder: Processes 3D brain scans
   - Clinical Encoder: Handles demographic and current score data
   - Fusion Regressor: Combines features for final predictions

## 👥 Contributors

| Student ID | Name |
|------------|------|
| | Phan Văn Hiếu (Phan Van Hieu) |
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
