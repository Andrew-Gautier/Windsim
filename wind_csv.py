import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('frt_wind.csv')

print(df.head())

# Set the plot style
sns.set(style='whitegrid')

# Line plot for the electrical signals over time
def plot_signals_over_time(df):
    plt.figure(figsize=(12, 6))
    for column in df.columns[1:]:  # Assuming the first column is time
        plt.plot(df.iloc[:, 0], df[column], label=column)
    plt.title('Electrical Signals Over Time')
    plt.xlabel('Time')
    plt.ylabel('Signal Value')
    plt.legend()
    plt.show()


threephase = pd.read_csv('wind_profile.csv')
two_phase = pd.read_csv('frt_wind.csv')

plot_signals_over_time(threephase)
plot_signals_over_time(two_phase)