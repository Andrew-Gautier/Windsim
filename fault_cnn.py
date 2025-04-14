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

def prepare_datasets(csv_path, label_column, window_size=100, overlap_size=50, 
                     test_size=0.2, val_size=0.2, random_state=42):
    df = pd.read_csv(csv_path)
    features = df.drop(label_column, axis=1).values
    labels = df[label_column].values

    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels)
    
    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val)
    
    # Create datasets
    train_dataset = WindFaultDataset(X_train, y_train, window_size, overlap_size)
    val_dataset = WindFaultDataset(X_val, y_val, window_size, overlap_size)
    test_dataset = WindFaultDataset(X_test, y_test, window_size, overlap_size)
    
    return train_dataset, val_dataset, test_dataset



class WindFaultCNN(nn.Module):

    def __init__(self, input_channels, num_classes, window_size):

        super(WindFaultCNN, self).__init__()
        # Dynamic kernel sizing
        kernel_size = min(3, window_size)
        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=kernel_size, padding='same')
        self.relu = nn.ReLU()
        # Adaptive pooling instead of fixed pooling
        self.pool = nn.AdaptiveMaxPool1d(output_size=window_size//2)
        # Second convolution with dynamic kernel
        self.conv2 = nn.Conv1d(16, 32, kernel_size=kernel_size, padding='same')
        # Calculate size after convolutions and pooling
        self.fc1_input_size = 32 * (window_size // 2)
        # Fully connected layers
        self.fc1 = nn.Linear(self.fc1_input_size, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
    
def train_model(model, train_loader, val_loader, num_epochs, optimizer, class_weights, device='cpu'):
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Track losses
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curves')
    plt.legend()
    plt.show()
    
    print("Training complete.")
    
def validate_model(model, val_loader, class_weights, device='cpu'):
    model.eval()
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy    
        
def test_model(model, test_loader, criterion, num_classes):
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

    # Suggestions for future work: 
    # 1.  Adding batch normalization layers after each convolution can help stabilize and accelerate training by normalizing the feature maps. 
    # 2.  To avoid overfitting, especially in fully connected layers, adding dropout (e.g., with a 20-50% rate) could be helpful. 
    # 3.  Strided convolutions instead of pooling could allow the model to better downsample.
    # 4.  Deeper architecture or simple residual connections may help the model extract more complex features.
    # 5.  More importantly, data normalization is one of the most critical factors that can affect the training procedure. 
    # 6.  Instead of a constant learning rate, a learning rate scheduler (like ReduceLROnPlateau) can be used.