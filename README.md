# Brain Score Project 🧠

A deep learning project for predicting future cognitive test scores from brain MRI images and clinical data.
Link to report: https://docs.google.com/document/d/14LXy6imsjAijm7Upmz41WT-_iGXutofiIkkS5eesSwI/edit?usp=sharing

Link to model:
- r3d_18(6-12): https://www.kaggle.com/code/quyennam/r3d-18-model-6-12
- r3d_18(6-18): https://www.kaggle.com/code/trananh9804/r3d-18-model-6-18
- SWinUNETR-Interaction(6-18): https://www.kaggle.com/code/phnvnh/swinunetr-interactions

## 📋 Overview

This project aims to predict three cognitive test scores (ADAS11, ADAS13, MMSCORE).

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
│   │   ├── encoders.py      # MRI encoders
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
├── notebooks/             # Jupyter notebooks for kaggle training
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
     - Current diagnosis
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
| 22022527 | Phan Văn Hiếu (Phan Van Hieu) |
| 22022533 | Nguyễn Đức Minh (Nguyen Duc Minh) |
| 22022594 | Trần Tiến Nam (Tran Tien Nam) |
| 22022628 | Vũ Đình Quang Huy (Vu Dinh Quang Huy) |
| 22022603 | Nguyễn Trọng Khánh (Nguyen Trong Khanh) |
| 22022574 | Bùi Văn Khải (Bui Van Khai) |

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
