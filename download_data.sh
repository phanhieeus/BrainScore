#!/bin/bash

# Install gdown if not already installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Create data directory if it doesn't exist
mkdir -p data

# Download the file from Google Drive
echo "Downloading data from Google Drive..."
gdown "https://drive.google.com/uc?id=1aJvY6XzqqM6yaSPsgTLTMWCoxaMpHhNC" -O data/brain_score_data.zip

# Unzip the file
echo "Extracting data..."
unzip -q data/brain_score_data.zip -d data/

# Remove the zip file
echo "Cleaning up..."
rm data/brain_score_data.zip

echo "Done! Data has been downloaded and extracted to the data/ directory." 