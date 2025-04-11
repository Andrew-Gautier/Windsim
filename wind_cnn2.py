# This file is to store the CNN model for wind fault prediction so it can be easily imported into jupyter notebooks for experimentation

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot

class WindFaultDataset(Dataset):
    """ PyTorch Dataset for Wind Fault Classification from single CSV file """
    def __init__(self, data, labels, window_size, overlap_size):
        self.data = []
        self.labels = []
        self.scaler = MinMaxScaler()
        
        # Normalize the data
        feature_data = self.scaler.fit_transform(data)
        
        # Create rolling windows
        windows, window_labels = self.create_rolling_windows(feature_data, labels, window_size, overlap_size)
        self.data = windows
        self.labels = window_labels

    def create_rolling_windows(self, data, labels, window_size, overlap_size):
        """ Generate overlapping rolling windows of data. """
        windows = []
        window_labels = []
        step_size = window_size - overlap_size
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data[i:i + window_size]
            # Use the label of the last point in the window
            label = labels[i + window_size - 1]
            windows.append(window)
            window_labels.append(label)
            
        return np.array(windows), np.array(window_labels)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def prepare_datasets(csv_path, label_column, window_size=100, overlap_size=50, test_size=0.2, random_state=42):
    """
    Prepare train and test datasets from a single CSV file.
    
    Args:
        csv_path: Path to the CSV file
        label_column: Name of the column containing labels
        window_size: Size of each window
        overlap_size: Overlap between consecutive windows
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        train_dataset, test_dataset: WindFaultDataset objects for training and testing
    """
    # Load the data
    df = pd.read_csv(csv_path)
    
    # Separate features and labels
    features = df.drop(label_column, axis=1).values
    labels = df[label_column].values
    
    # Split into train and test sets (stratified by label)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    # Create datasets
    train_dataset = WindFaultDataset(X_train, y_train, window_size, overlap_size)
    test_dataset = WindFaultDataset(X_test, y_test, window_size, overlap_size)
    
    return train_dataset, test_dataset

class WindFaultCNN(nn.Module):
    def __init__(self, input_channels, num_classes, window_size):
        super(WindFaultCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * (window_size // 2 // 2), 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Change shape to (batch, features, time)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    
def train_model(model, train_loader, num_epochs, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {train_loss/len(train_loader)}")
    print("Training complete.")
        
def evaluate_model(model, test_loader, criterion, num_classes):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():  # No gradient calculation
        for inputs, labels in test_loader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predicted classes
            _, preds = torch.max(outputs, 1)  # Convert softmax to class indices

            # Store results
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, digits=4)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Test Loss: {total_loss / len(test_loader):.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:\n", class_report)
    print("\nConfusion Matrix:\n", conf_matrix)
    
    # Compute AUC-ROC
    all_labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
    all_probs = np.array(all_probs)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot AUC-ROC curve
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()

    return accuracy, class_report, conf_matrix, roc_auc