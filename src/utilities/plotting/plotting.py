# src/utilities/plotting/plotting.py

import os
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from src.utilities.plotting.animate_model import animate_model
from src.utilities.locations import Locations

def check_remote_and_animate(remote, path_to_shape_model_file, plotted_variable_array, rotation_axis, 
                           sunlight_direction, timesteps_per_day, solar_distance_au, rotation_period_hr, emissivity, **kwargs):
    """
    Handles animation logic based on whether remote mode is enabled.
    If remote is False, calls animate_model function directly.
    If remote is True, saves the arguments into .npz and .json files.
    """
    if not remote:
        # Local mode: call animate_model as usual
        # Pop apply_kernel_based_roughness flag if provided
        apply_kernel_based_roughness = kwargs.pop('apply_kernel_based_roughness', False)
        # Extract dome radius factor for scaling canonical dome areas
        dome_radius_factor = kwargs.pop('dome_radius_factor', 1.0)
        # Extract required arguments from kwargs
        plot_title = kwargs.pop('plot_title', 'Temperature distribution')
        axis_label = kwargs.pop('axis_label', 'Temperature (K)')
        animation_frames = kwargs.pop('animation_frames', 200)
        save_animation = kwargs.pop('save_animation', False)
        save_animation_name = kwargs.pop('save_animation_name', 'temperature_animation.gif')
        background_colour = kwargs.pop('background_colour', 'black')
        colour_map = kwargs.pop('colour_map', 'coolwarm')
        
        # Call animate_model with all required arguments, including roughness flag
        animate_model(path_to_shape_model_file, plotted_variable_array, rotation_axis, sunlight_direction,
                     timesteps_per_day, solar_distance_au, rotation_period_hr, emissivity,
                     plot_title, axis_label, animation_frames, save_animation, save_animation_name,
                     background_colour, dome_radius_factor=dome_radius_factor, colour_map=colour_map,
                     apply_kernel_based_roughness=apply_kernel_based_roughness,
                     animation_debug_mode=False)
    else:
        # Remote mode: create or reuse a directory to save the animation parameters
        # Allow callers to pass an explicit output directory to keep artifacts together
        provided_output_dir = kwargs.pop('output_dir', None)
        if provided_output_dir is not None:
            output_dir = provided_output_dir
            os.makedirs(output_dir, exist_ok=True)
        else:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            loc = Locations()
            output_dir = os.path.join(loc.remote_outputs, f"animation_outputs_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

        # Save numpy arrays to NPZ file
        numpy_arrays = {
            'plotted_variable_array': plotted_variable_array,
            'rotation_axis': rotation_axis,
            'sunlight_direction': sunlight_direction
        }
        npz_file = os.path.join(output_dir, 'animation_params.npz')
        np.savez_compressed(npz_file, **numpy_arrays)

        # Pop apply_kernel_based_roughness flag if provided
        apply_kernel_based_roughness = kwargs.pop('apply_kernel_based_roughness', False)

        if apply_kernel_based_roughness:
            print(f'Roughness detected - new logic applied\n')

            print(f'Plotted variable array looks like {len(plotted_variable_array)}')

        else: 
            print(f'Smooth model detected\n')

            print(f'Plotted variable array looks like {len(plotted_variable_array)}')


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
                'emissivity': emissivity,   # Save emissivity for animate_model
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
