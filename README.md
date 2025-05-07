# Brain Cognitive Score Prediction using MONAI

This project uses MONAI framework to predict cognitive scores (ADAS11, ADAS13, MMSCORE, CDGLOBAL) from 3D T1 MRI brain images and demographic data.

## Project Structure

```
.
├── data/
│   ├── T1_biascorr_brain_data/    # Raw MRI images
│   ├── c1_c2_cognitive_score.csv  # Cognitive scores
│   └── c1_c2_demographics.csv     # Demographic data
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # Custom dataset class
│   │   └── preprocessing.py      # Data preprocessing utilities
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py             # Model architecture
│   │   └── grad_cam.py          # Grad-CAM implementation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py           # Training utilities
│   │   └── metrics.py           # Evaluation metrics
│   └── utils/
│       ├── __init__.py
│       └── visualization.py      # Visualization utilities
├── notebooks/
│   └── analysis.ipynb           # Analysis and visualization notebooks
├── configs/
│   └── config.yaml              # Configuration file
├── requirements.txt             # Project dependencies
└── README.md                    # Project documentation
```

## Environment Setup

### 1. Install Python and Virtual Environment Tools

#### For Ubuntu/Debian:
```bash
# Install Python and pip
sudo apt update
sudo apt install python3 python3-pip python3-venv

# Install virtualenv (optional but recommended)
pip3 install virtualenv
```

#### For Windows:
1. Download and install Python from [python.org](https://www.python.org/downloads/)
2. During installation, make sure to check "Add Python to PATH"
3. Open Command Prompt and install virtualenv:
```bash
pip install virtualenv
```

### 2. Create and Activate Virtual Environment

#### Using venv (Built-in):
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Ubuntu/Debian:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

#### Using virtualenv:
```bash
# Create virtual environment
virtualenv venv

# Activate virtual environment
# On Ubuntu/Debian:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Verify CUDA availability (if using GPU)
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

## Data Setup

1. Create required directories:
```bash
mkdir -p data/T1_biascorr_brain_data
```

2. Place your data files:
- MRI images in `data/T1_biascorr_brain_data/`
- Cognitive scores CSV in `data/c1_c2_cognitive_score.csv`
- Demographics CSV in `data/c1_c2_demographics.csv`

## Usage

### 1. Data Preprocessing
```bash
# Activate virtual environment if not already activated
source venv/bin/activate  # On Ubuntu/Debian
# or
venv\Scripts\activate  # On Windows

# Run preprocessing
python src/data/preprocessing.py
```

### 2. Model Training
```bash
# Train the model
python src/main.py
```

### 3. Model Evaluation and Visualization
```bash
# Generate visualizations
python src/utils/visualization.py
```

## Features

- Multi-input model combining 3D MRI images and demographic data
- Cognitive score prediction (ADAS11, ADAS13, MMSCORE, CDGLOBAL)
- Longitudinal analysis of cognitive decline
- Model interpretation using Grad-CAM
- Comprehensive evaluation metrics

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- See requirements.txt for full list of dependencies

## Troubleshooting

### Common Issues

1. **CUDA not available:**
   - Check NVIDIA drivers are installed
   - Verify CUDA toolkit installation
   - Try running with CPU: modify `config.yaml` to use `device: "cpu"`

2. **Memory issues:**
   - Reduce batch size in `config.yaml`
   - Use smaller image size
   - Enable gradient checkpointing

3. **Package conflicts:**
   - Create fresh virtual environment
   - Install dependencies in order: `pip install -r requirements.txt`

### Getting Help

If you encounter any issues:
1. Check the error message carefully
2. Verify your environment setup
3. Ensure all dependencies are correctly installed
4. Check GPU availability and memory
5. Review the configuration in `config.yaml` 