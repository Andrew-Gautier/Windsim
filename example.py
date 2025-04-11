param_grid = {
    'window_size': [5, 10, 15, 20],  # Test smaller window sizes
    'overlap_size': [0, 2, 5],        # Relative to window size
    'learning_rate': [0.001, 0.005],
    'batch_size': [32, 64],
    'num_epochs': [30, 50],          # Increased epochs for smaller windows
    'optimizer': ['Adam']
}


best_accuracy = 0
best_params = {}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameter search loop
for params in itertools.product(*param_grid.values()):
    window_size, overlap_size, lr, batch_size, num_epochs, optimizer_name = params
    print(f"\nTesting params: {params}")
    
    try:
        # Prepare datasets
        train_dataset, val_dataset, test_dataset = prepare_datasets(
            csv_path='combined_balanced_dataset.csv',
            label_column='Fault_Type',
            window_size=window_size,
            overlap_size=overlap_size,
            test_size=0.2,
            val_size=0.25
        )
        
        # Calculate class weights
        class_counts = np.bincount(train_dataset.labels)
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        class_weights = torch.tensor(total_samples / (num_classes * class_counts), dtype=torch.float32)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        num_features = train_dataset.data.shape[2]
        model = WindFaultCNN(
            input_channels=num_features,
            num_classes=num_classes,
            window_size=window_size
        ).to(device)

        # Select optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr)

        # Train and validate
        #train_model(model, train_loader, num_epochs, optimizer, class_weights, device)
        val_accuracy = validate_model(model, val_loader, class_weights, device)
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = params
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"New best accuracy: {best_accuracy:.4f}")

    except Exception as e:
        print(f"Error with params {params}: {str(e)}")
        continue

# Evaluate best model on test set
print(f"\nBest parameters: {best_params}")
train_dataset, val_dataset, test_dataset = prepare_datasets(
    csv_path='combined_balanced_dataset.csv',
    label_column='Fault_Type',
    window_size=best_params[0],
    overlap_size=best_params[1]
)

test_loader = DataLoader(test_dataset, batch_size=best_params[3], shuffle=False)
best_model = WindFaultCNN(
    input_channels=train_dataset.data.shape[2],
    num_classes=len(np.unique(train_dataset.labels)),
    window_size=best_params[0]
).to(device)
best_model.load_state_dict(torch.load('best_model.pth'))

class_counts = np.bincount(train_dataset.labels)
class_weights = torch.tensor(class_counts.sum() / (len(class_counts) * class_counts), dtype=torch.float32)
# Note that evaluate model is deprecated, now use test_model
final_accuracy, report, conf_matrix, roc_auc = evaluate_model(
    best_model, test_loader, 
    nn.CrossEntropyLoss(weight=class_weights.to(device)),
    len(class_counts)
)

print(f"Final Test Accuracy: {final_accuracy:.4f}")
print(report)