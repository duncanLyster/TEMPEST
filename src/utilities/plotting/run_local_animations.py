import json
import numpy as np
import os
from datetime import datetime

from animate_model import animate_model

def get_output_folders(base_dir):
    """Retrieve all folders in the base directory sorted by modification time."""
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    # Sort by modification time, newest first
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(base_dir, f)), reverse=True)
    return folders

def get_user_confirmation(default_folder, available_folders):
    """Ask the user if they want to use the default most recent folder or choose another."""
    print(f"The most recent folder is: {default_folder}")
    use_default = input("Do you want to use this folder? (y/n): ").strip().lower()

    if use_default == 'y':
        return default_folder
    else:
        print("\nAvailable folders:")
        for i, folder in enumerate(available_folders):
            print(f"{i}: {folder}")
        
        # Get the user's choice
        while True:
            try:
                choice = int(input("Select a folder by number: "))
                if 0 <= choice < len(available_folders):
                    return available_folders[choice]
                else:
                    print(f"Invalid selection. Please select a number between 0 and {len(available_folders) - 1}.")
            except ValueError:
                print("Please enter a valid number.")

def run_saved_animation(json_file, npz_file):
    """
    Loads saved animation parameters from JSON and NPZ files and runs the animation.
    
    Parameters:
    - json_file (str): Path to the JSON file containing non-NumPy parameters.
    - npz_file (str): Path to the NPZ file containing NumPy arrays.
    """
    # Load data from files
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    npz_data = np.load(npz_file, allow_pickle=True)
    
    # Extract base arguments
    args = [
        json_data['args'][0],  # path_to_shape_model_file
        npz_data['plotted_variable_array'],  # plotted_variable_array
        npz_data['rotation_axis'],  # rotation_axis
        npz_data['sunlight_direction'],  # sunlight_direction
    ]
    
    # Extract required positional arguments
    kwargs = json_data['kwargs']
    args.extend([
        kwargs.pop('timesteps_per_day'),
        kwargs.pop('solar_distance_au'),
        kwargs.pop('rotation_period_hr')
    ])
    
    # Remove placeholder entries from kwargs
    kwargs.pop('rotation_axis', None)
    kwargs.pop('sunlight_direction', None)
    
    # Print debugging information
    print("\nLoaded parameters:")
    print(f"Positional args count: {len(args)}")
    print(f"Keyword args: {list(kwargs.keys())}\n")
    
    # Call the animate_model function
    print("Running animation...")
    animate_model(*args, **kwargs)
    print("Animation complete.")
    
    # Print debugging information
    print("Loaded arguments:")
    print(f"Positional args count: {len(args)}")
    print(f"Keyword args: {list(kwargs.keys())}")
    
    # Call the animate_model function with the correct arguments
    print(f"Running animation with parameters from {json_file} and {npz_file}...")
    animate_model(*args, **kwargs)
    print("Animation complete.")


def main():
    # Set the base directory for output folders
    base_dir = 'outputs/remote_outputs/'
    
    # Check if base directory exists
    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Exiting.")
        return
    
    # Get all available folders
    available_folders = get_output_folders(base_dir)
    
    if not available_folders:
        print(f"No folders found in {base_dir}. Exiting.")
        return
    
    # The most recent folder
    default_folder = available_folders[0]

    # Ask the user for confirmation or selection
    selected_folder = get_user_confirmation(default_folder, available_folders)

    # Set the file paths
    json_file = os.path.join(base_dir, selected_folder, 'animation_params.json')
    npz_file = os.path.join(base_dir, selected_folder, 'animation_params.npz')

    # Check if both files exist
    if not os.path.exists(json_file) or not os.path.exists(npz_file):
        print(f"Required files not found in {selected_folder}. Exiting.")
        return
    
    # Run the saved animation
    run_saved_animation(json_file, npz_file)

if __name__ == "__main__":
    main()
