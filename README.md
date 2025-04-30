# Grid Fault Detection using CNN

This project implements a Convolutional Neural Network (CNN) for fault detection in energy grid using time-series sensor data. The model processes rolling windows of normalized sensor data to classify different fault conditions. There were four faults considered here: 3-phase, 2-phase, 2-phase-earth, and 1phase. For a more detailed explanation of the project see the final report pdf. 

## Key Features

- **Data Processing**: 
  - Rolling window generation with configurable window size and overlap
  - Min-Max normalization of sensor data
  - Stratified train/validation/test splits

- **CNN Architecture**:
  - 1D convolutional layers with dynamic kernel sizing
  - Adaptive max pooling
  - Two fully connected layers with ReLU activation
  - Softmax output for multi-class classification

- **Training & Evaluation**:
  - Cross-entropy loss with class weighting
  - Training progress visualization
  - Comprehensive evaluation metrics:
    - Accuracy
    - Classification report
    - Confusion matrix
    - ROC-AUC curves

## Requirements

- Python 3.7+
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- torchviz (for model visualization)

## Installation

All requirements to run this code exist in requirements.txt

If you are new to python, the best way to get started is:
1. Download Python. (Make sure you check add to PATH or environment variables).
2. Create a folder on your desktop.
3. Clone/Fork this repository there. 
4. Open it in VSCode.
5. In your terminal, (must be where your folder is) run python -m venv venv.
6. After you see the venv folder created, if windows, run venv/scripts/activate . For mac/linux should be source venv/bin/activate.
7. Then run pip install -r requirements.txt and watch it download all the necessary python packages.

