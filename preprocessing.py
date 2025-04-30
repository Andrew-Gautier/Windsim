import os
import pandas as pd
import numpy as np
from collections import defaultdict

def count_labels(csv_path, label_col='Fault_Type'):
    """
    Standalone function to count rows for each label in a CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        label_col (str): Name of label column (default 'Fault_Type')
    
    Returns:
        dict: Counts of each label {label: count}
    """
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")
    
    return df[label_col].value_counts().to_dict()

def extract_fault_type(filename):
    """
    Extract fault type from filename, handling different formats.
    Examples:
    - "2P.EARTH-60-20-0.csv" → 2
    - "1P-100-0-0.csv" → 1
    - "0.csv" (normal) → 0
    """
    # Remove extension
    base = os.path.splitext(filename)[0]
    
    # Try to extract the first number before any non-digit character
    for char in base:
        if char.isdigit():
            return int(char)
        elif char in ['P', '.']:
            continue
        else:
            break
    return 0  # Default to 0 if no number found

def combine_and_balance_csvs(input_dir, output_file, normal_csv_path, noise_level=0.02, label_col='Fault_Type'):
    """
    Combine CSV files into one balanced dataset with normal samples inserted between faults.
    
    Args:
        input_dir (str): Directory containing fault CSV files
        output_file (str): Path to save combined CSV
        normal_csv_path (str): Path to normal operation CSV file
        noise_level (float): Standard deviation for Gaussian noise
        label_col (str): Name of label column (default 'Fault_Type')
    """
    # Load normal data
    normal_df = pd.read_csv(normal_csv_path)
    
    # Collect all fault files and group by class
    fault_files = defaultdict(list)
    for fname in os.listdir(input_dir):
        if fname.endswith('.csv') and not fname.startswith('noisy-') and fname != os.path.basename(normal_csv_path):
            fault_type = extract_fault_type(fname)
            fault_files[fault_type].append(os.path.join(input_dir, fname))
    
    # Determine max samples per class
    max_samples = max(len(files) for files in fault_files.values()) if fault_files else 0
    
    # Prepare to collect all data
    combined_data = []
    
    # For each class, add samples until we reach max_samples
    for fault_type, files in fault_files.items():
        # Calculate how many samples we need to add for this class
        needed = max_samples - len(files)
        
        # Add original files
        for filepath in files:
            df = pd.read_csv(filepath)
            combined_data.append(df)
            
            # Add normal sample with noise after each fault (protecting label column)
            noisy_normal = normal_df.copy()
            numeric_cols = [col for col in noisy_normal.select_dtypes(include=[np.number]).columns 
                          if col != label_col]
            noisy_normal[numeric_cols] = noisy_normal[numeric_cols] + np.random.normal(0, noise_level, noisy_normal[numeric_cols].shape)
            combined_data.append(noisy_normal)
        
        # If we need more samples for this class, duplicate existing files with added noise
        for i in range(needed):
            # Select a file to duplicate (cycling through available files)
            file_to_dup = files[i % len(files)]
            df = pd.read_csv(file_to_dup)
            
            # Add noise to create a new sample (protecting label column)
            noisy_df = df.copy()
            numeric_cols = [col for col in noisy_df.select_dtypes(include=[np.number]).columns 
                          if col != label_col]
            noisy_df[numeric_cols] = noisy_df[numeric_cols] + np.random.normal(0, noise_level, noisy_df[numeric_cols].shape)
            combined_data.append(noisy_df)
            
            # Add normal sample with noise after each fault
            noisy_normal = normal_df.copy()
            noisy_normal[numeric_cols] = noisy_normal[numeric_cols] + np.random.normal(0, noise_level, noisy_normal[numeric_cols].shape)
            combined_data.append(noisy_normal)
    
    # Combine all dataframes and ensure label column is integer type
    final_df = pd.concat(combined_data, ignore_index=True)
    if label_col in final_df.columns:
        final_df[label_col] = final_df[label_col].astype(int)
    
    # Save the combined dataset
    final_df.to_csv(output_file, index=False)
    print(f"Saved balanced combined dataset to {output_file}")
    
    # Print label counts
    label_counts = count_labels(output_file, label_col)
    print("\nLabel counts in final dataset:")
    for label, count in sorted(label_counts.items()):
        print(f"Label {label}: {count} samples")
    
    # Also create a noisy version of the combined dataset (protecting label column)
    noisy_dir = os.path.dirname(output_file)
    os.makedirs(noisy_dir, exist_ok=True)
    noisy_output = os.path.join(noisy_dir, "noisy_" + os.path.basename(output_file))
    
    noisy_final_df = final_df.copy()
    numeric_cols = [col for col in noisy_final_df.select_dtypes(include=[np.number]).columns 
                   if col != label_col]
    noisy_final_df[numeric_cols] = noisy_final_df[numeric_cols] + np.random.normal(0, noise_level, noisy_final_df[numeric_cols].shape)
    noisy_final_df.to_csv(noisy_output, index=False)
    print(f"\nAlso created noisy version at {noisy_output}")
    
    # Print noisy version label counts
    noisy_label_counts = count_labels(noisy_output, label_col)
    print("\nLabel counts in noisy dataset:")
    for label, count in sorted(noisy_label_counts.items()):
        print(f"Label {label}: {count} samples")

def add_noise_to_csv_files(input_dir, output_subdir, noise_level=0.02):
    """
    Adds noise to data from CSV files in a directory and saves them as new CSV files in a subfolder.

    Parameters:
    - input_dir (str): Path to the directory containing the original CSV files.
    - output_subdir (str): Name of the subfolder to save the augmented CSV files.
    - noise_level (float): Standard deviation of the Gaussian noise to be added.
    """
    # Create the output subdirectory if it doesn't exist
    output_dir = os.path.join(input_dir, output_subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all CSV files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            file_path = os.path.join(input_dir, file_name)
            df = pd.read_csv(file_path)
            
            # Add Gaussian noise to the data
            noise = np.random.normal(0, noise_level, df.shape)
            df_noisy = df + noise
            
            # Save the augmented data to a new CSV file in the output subdirectory
            output_file_path = os.path.join(output_dir, f"noisy-{file_name}")
            df_noisy.to_csv(output_file_path, index=False)
            print(f"Saved noisy data to {output_file_path}")






# Example usage:
if __name__ == "__main__":
    input_directory = "Distribution_faults"
    normal_csv = "normal.csv"  # Path to your normal operation CSV
    output_path = "combined_balanced_dataset.csv"
    
    # First demonstrate the count_labels function
    test_file = "combined_balanced_dataset.csv"  # Example file
    if os.path.exists(test_file):
        print(f"\nTesting count_labels on {test_file}:")
        print(count_labels(test_file))
    
    # Run the main function
    combine_and_balance_csvs(input_directory, output_path, normal_csv)