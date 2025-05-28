# Brain Score Project 🧠

A deep learning project for predicting future cognitive test scores from brain MRI images and demographics data.

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

## 🚀 Quick Start

For detailed setup and usage instructions, please refer to [GETTING_STARTED.md](GETTING_STARTED.md).

## 📁 Project Structure

```
BrainScoreProject/
├── data/                      # Data directory
│   ├── T1_biascorr_brain_data/  # MRI images
│   ├── c1_c2_cognitive_score.csv # Cognitive test scores
│   ├── c1_c2_demographics.csv    # Demographics data
│   ├── single_test_points.csv    # Processed data
│   ├── train_data.csv           # Training set
│   └── test_data.csv            # Test set
├── src/                      # Source code
│   ├── models/               # Model definitions
│   ├── create_single_test_dataset.py  # Create dataset
│   ├── split_data.py         # Split data
│   ├── dataset.py            # Dataset class
│   └── train.py              # Training script
├── logs/                     # Training logs
├── checkpoints/              # Model checkpoints
└── requirements.txt          # Required packages
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
| | Nguyễn Khánh (Nguyen Khanh) |
| | Nguyễn Khải (Nguyen Khai) |

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
