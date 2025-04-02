import pandas as pd # type: ignore
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


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