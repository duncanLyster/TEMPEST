# src/utilities/plotting/plotting.py

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from src.utilities.plotting.animate_model import animate_model

def check_remote_and_animate(remote, path_to_shape_model_file, plotted_variable_array, rotation_axis, 
                           sunlight_direction, timesteps_per_day, solar_distance_au, rotation_period_hr, 
                           **kwargs):
    """
    Handles animation logic based on whether remote mode is enabled.
    If remote is False, calls animate_model function directly.
    If remote is True, saves the arguments into .npz and .json files.
    """
    if not remote:
        # Local mode: call animate_model as usual
        animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction,
                     timesteps_per_day, solar_distance_au, rotation_period_hr, **kwargs)
    else:
        # Remote mode: create a directory to save the animation parameters
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"outputs/remote_outputs/animation_outputs_{timestamp}" # TODO: include in locations class
        os.makedirs(output_dir, exist_ok=True)

        # Save numpy arrays to NPZ file
        numpy_arrays = {
            'plotted_variable_array': plotted_variable_array,
            'rotation_axis': rotation_axis,
            'sunlight_direction': sunlight_direction
        }
        npz_file = os.path.join(output_dir, 'animation_params.npz')
        np.savez_compressed(npz_file, **numpy_arrays)

        # Save other parameters to JSON file
        json_params = {
            'args': [
                path_to_shape_model_file,  # First argument is always the shape model file
                None,  # Placeholder for plotted_variable_array (stored in NPZ)
            ],
            'kwargs': {
                # Required positional arguments stored as kwargs for easier handling
                'timesteps_per_day': timesteps_per_day,
                'solar_distance_au': solar_distance_au,
                'rotation_period_hr': rotation_period_hr,
                'rotation_axis': None,  # Placeholder for rotation_axis (stored in NPZ)
                'sunlight_direction': None,  # Placeholder for sunlight_direction (stored in NPZ)
                # Additional keyword arguments
                **kwargs
            }
        }

        json_file = os.path.join(output_dir, 'animation_params.json')
        with open(json_file, 'w') as f:
            json.dump(json_params, f, indent=2)

        print(f"Animation parameters saved in:\nJSON: {json_file}\nNPZ: {npz_file}")
