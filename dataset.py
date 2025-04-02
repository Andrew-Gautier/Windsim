import os
import pandas as pd
import numpy as np
from preprocessing import add_noise_to_csv_files
from collections import defaultdict

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

def combine_and_balance_csvs(input_dir, output_file, normal_csv_path, noise_level=0.02):
    """
    Combine CSV files into one balanced dataset with normal samples inserted between faults.
    
    Args:
        input_dir (str): Directory containing fault CSV files
        output_file (str): Path to save combined CSV
        normal_csv_path (str): Path to normal operation CSV file
        noise_level (float): Standard deviation for Gaussian noise
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
            
            # Add normal sample with noise after each fault
            noisy_normal = normal_df.copy()
            numeric_cols = noisy_normal.select_dtypes(include=[np.number]).columns
            noisy_normal[numeric_cols] = noisy_normal[numeric_cols] + np.random.normal(0, noise_level, noisy_normal[numeric_cols].shape)
            combined_data.append(noisy_normal)
        
        # If we need more samples for this class, duplicate existing files with added noise
        for i in range(needed):
            # Select a file to duplicate (cycling through available files)
            file_to_dup = files[i % len(files)]
            df = pd.read_csv(file_to_dup)
            
            # Add noise to create a new sample (only to numeric columns)
            noisy_df = df.copy()
            numeric_cols = noisy_df.select_dtypes(include=[np.number]).columns
            noisy_df[numeric_cols] = noisy_df[numeric_cols] + np.random.normal(0, noise_level, noisy_df[numeric_cols].shape)
            combined_data.append(noisy_df)
            
            # Add normal sample with noise after each fault
            noisy_normal = normal_df.copy()
            noisy_normal[numeric_cols] = noisy_normal[numeric_cols] + np.random.normal(0, noise_level, noisy_normal[numeric_cols].shape)
            combined_data.append(noisy_normal)
    
    # Combine all dataframes
    final_df = pd.concat(combined_data, ignore_index=True)
    
    # Save the combined dataset
    final_df.to_csv(output_file, index=False)
    print(f"Saved balanced combined dataset to {output_file}")
    
    # Also create a noisy version of the combined dataset
    noisy_dir = os.path.dirname(output_file)
    os.makedirs(noisy_dir, exist_ok=True)
    noisy_output = os.path.join(noisy_dir, "noisy_" + os.path.basename(output_file))
    
    # Create noisy version by adding noise to numeric columns
    noisy_final_df = final_df.copy()
    numeric_cols = noisy_final_df.select_dtypes(include=[np.number]).columns
    noisy_final_df[numeric_cols] = noisy_final_df[numeric_cols] + np.random.normal(0, noise_level, noisy_final_df[numeric_cols].shape)
    noisy_final_df.to_csv(noisy_output, index=False)
    print(f"Also created noisy version at {noisy_output}")

# Example usage:
if __name__ == "__main__":
    input_directory = "Distribution_faults"
    normal_csv = "Distribution_faults\\normal-.csv"  # Path to your normal operation CSV
    output_path = "combined_balanced_dataset.csv"
    
    combine_and_balance_csvs(input_directory, output_path, normal_csv)