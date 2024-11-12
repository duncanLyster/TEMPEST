import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_csv_files(folder_path, normalize=False):
    """
    Read all CSV files from a folder, optionally normalize their data, and plot on the same axes.
    Assumes each CSV has exactly two columns.
    
    Args:
        folder_path (str): Path to the folder containing CSV files
        normalize (bool): If True, normalize y-values to a maximum of 1
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get list of CSV files in the folder
    csv_files = list(Path(folder_path).glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {folder_path}")
        return
    
    # Plot each CSV file
    for csv_file in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(csv_file)
            
            # Check if the file has exactly 2 columns
            if len(df.columns) != 2:
                print(f"Skipping {csv_file.name} - Expected 2 columns, found {len(df.columns)}")
                continue
            
            # Normalize y-values if the flag is set to True
            if normalize:
                max_val = df.iloc[:, 1].abs().max()
                if max_val != 0:
                    df.iloc[:, 1] = df.iloc[:, 1] / max_val
            
            # Plot the data
            ax.plot(df.iloc[:, 0], df.iloc[:, 1], label=csv_file.stem)
            
        except Exception as e:
            print(f"Error processing {csv_file.name}: {str(e)}")
    
    # Customize the plot
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(df.columns[1] + (' (Normalized)' if normalize else ''))
    ax.set_title('Combined Plot of CSV Data' + (' (Normalized)' if normalize else ''))
    ax.grid(True)
    ax.legend()
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Show the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Replace with your folder path and set normalize flag
    folder_path = "/Users/duncan/Desktop/DPhil/TEMPEST/visible_phase_curve_data"
    plot_csv_files(folder_path, normalize=True)
