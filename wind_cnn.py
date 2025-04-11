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
import matplotlib.pyplot as plt
import seaborn as sns
from torchviz import make_dot

class WindFaultDataset(Dataset):
    """ PyTorch Dataset for Wind Fault Classification """
    def __init__(self, file_paths, label_mapping, window_size, overlap_size):
        self.data = []
        self.labels = []
        self.label_mapping = {}
        self.scaler = MinMaxScaler()

        for file_path in file_paths:
            features, label = self.preprocess(file_path, label_mapping)
            windows, labels = self.create_rolling_windows(features, label, window_size, overlap_size)
            self.data.append(windows)
            self.labels.append(labels)

         # Check if all windows have the same number of columns
        num_columns = [window.shape[2] for window in self.data]
        if len(set(num_columns)) != 1:
            raise ValueError(f"Mismatch in number of columns: {num_columns}")

        self.data = np.vstack(self.data)  # Stack all windows
        self.labels = np.hstack(self.labels)

    def preprocess(self, file_path, label_mapping):
        """ Load and preprocess a CSV file. """
        df = pd.read_csv(file_path)
  
        df = df.drop(df.columns[0], axis=1)  # Dropping time column
        df = df.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
        
        # Extract label from filename
        fault_type = os.path.basename(file_path).split("-")[0]
        label = label_mapping.get(fault_type, 0)
        
        # Normalize features
        feature_data = self.scaler.fit_transform(df)
        # Add label column to feature_data
        label_column = np.full((feature_data.shape[0], 1), label)
        features = np.hstack((feature_data, label_column))
        
        return features, label

    def create_rolling_windows(self, data, label, window_size, overlap_size):
        #""" Generate overlapping rolling windows of data. """
        windows = []
        labels = []
        step_size = window_size - overlap_size
        for i in range(0, len(data) - window_size + 1, step_size):  # Overlapping windows
            windows.append(data[i:i + window_size])
            labels.append(label)
        return np.array(windows), np.array(labels)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

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

