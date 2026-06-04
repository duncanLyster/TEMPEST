import json
import numpy as np
import os

from animate_model import animate_model

# Map-style playback options
CENTER_MESH_AT_ORIGIN = True
ROTATE_MESH = False
COLOUR_MAP = "viridis"

# Set to None to animate all timesteps
# Set to an integer to display only a single timestep
SELECT_TIMESTEP = 28


def get_output_folders(base_dir):
    """Retrieve all folders in the base directory sorted by modification time."""
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folders.sort(key=lambda f: os.path.getmtime(os.path.join(base_dir, f)), reverse=True)
    return folders


def get_user_confirmation(prompt_name, default_folder, available_folders):
    """
    Ask the user if they want to use the default most recent folder or choose another.
    `prompt_name` is used to distinguish option 0 and option 1.
    """
    print(f"\nThe most recent folder for {prompt_name} is: {default_folder}")
    use_default = input(f"Do you want to use this folder for {prompt_name}? (y/n): ").strip().lower()

    if use_default == "y":
        return default_folder

    print("\nAvailable folders:")
    for i, folder in enumerate(available_folders):
        print(f"{i}: {folder}")

    while True:
        try:
            choice = int(input(f"Select a folder for {prompt_name} by number: "))
            if 0 <= choice < len(available_folders):
                return available_folders[choice]
            else:
                print(f"Invalid selection. Please select a number between 0 and {len(available_folders) - 1}.")
        except ValueError:
            print("Please enter a valid number.")


def get_optional_float(prompt_text):
    """
    Ask the user for an optional float.
    Press Enter to leave it unset.
    """
    while True:
        raw = input(prompt_text).strip()
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print("Please enter a valid number, or press Enter to leave it unset.")


def load_saved_animation(folder_path):
    """
    Load animation parameters from a saved output folder.
    Returns:
        json_data, npz_data
    """
    json_file = os.path.join(folder_path, "animation_params.json")
    npz_file = os.path.join(folder_path, "animation_params.npz")

    if not os.path.exists(json_file) or not os.path.exists(npz_file):
        raise FileNotFoundError(f"Required files not found in {folder_path}")

    with open(json_file, "r") as f:
        json_data = json.load(f)

    npz_data = np.load(npz_file, allow_pickle=True)
    return json_data, npz_data


def build_animation_args(json_data, npz_data, plotted_variable_array):
    """
    Build positional args and kwargs for animate_model from saved data.
    """
    args = [
        json_data["args"][0],               # path_to_shape_model_file
        plotted_variable_array,             # plotted_variable_array
        npz_data["rotation_axis"],          # rotation_axis
        npz_data["sunlight_direction"],     # sunlight_direction
    ]

    kwargs = dict(json_data["kwargs"])

    # Required positional scalar parameters
    args.extend([
        kwargs.pop("timesteps_per_day"),
        kwargs.pop("solar_distance_au"),
        kwargs.pop("rotation_period_hr"),
        kwargs.pop("emissivity", 0.5),
    ])

    plot_title = kwargs.pop("plot_title", "Difference plot: option 0 minus option 1")
    axis_label = kwargs.pop("axis_label", "Difference")
    animation_frames = kwargs.pop("animation_frames", 200)
    save_animation = kwargs.pop("save_animation", False)
    save_animation_name = kwargs.pop("save_animation_name", "difference_animation.gif")
    background_colour = kwargs.pop("background_colour", "black")

    args.extend([
        plot_title,
        axis_label,
        animation_frames,
        save_animation,
        save_animation_name,
        background_colour,
    ])

    # Remove placeholders if present
    kwargs.pop("rotation_axis", None)
    kwargs.pop("sunlight_direction", None)
    kwargs.pop("shape_model", None)

    return args, kwargs, animation_frames


def prepare_difference_array(diff, select_timestep, animation_frames):
    """
    If a timestep is selected, take that timestep and repeat it across all frames
    so animate_model never tries to index past a single column.
    """
    if select_timestep is None:
        return diff

    if diff.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {diff.shape}")

    if select_timestep < 0 or select_timestep >= diff.shape[1]:
        raise ValueError(
            f"SELECT_TIMESTEP={select_timestep} is outside valid range "
            f"[0, {diff.shape[1] - 1}] for the timestep axis."
        )

    print(f"Displaying timestep {select_timestep}")

    # Keep cells in rows, then repeat one selected timestep across all frames
    one_frame = diff[:, select_timestep:select_timestep + 1]
    repeated = np.repeat(one_frame, animation_frames, axis=1)

    return repeated


def apply_scale_limits(data, vmin, vmax):
    """
    Clip the plotted data to user-specified min/max values if provided.
    This forces the color scaling to stay within the requested range.
    """
    if vmin is None and vmax is None:
        return data

    clipped = np.array(data, copy=True)

    if vmin is not None and vmax is not None and vmin > vmax:
        raise ValueError(f"Scale minimum {vmin} is greater than maximum {vmax}.")

    if vmin is not None:
        clipped = np.maximum(clipped, vmin)
    if vmax is not None:
        clipped = np.minimum(clipped, vmax)

    return clipped

def inject_scale_sentinels(data, scale_min=None, scale_max=None):
    """
    Force the plotted array to contain the requested limits, so any auto-scaled
    colorbar sees the full range.

    This changes only a copy of the data passed to the animation.
    """
    arr = np.array(data, copy=True)

    if arr.size == 0:
        return arr

    if scale_min is not None:
        arr[0, 0] = scale_min

    if scale_max is not None:
        row = 1 if arr.shape[0] > 1 else 0
        arr[row, 0] = scale_max

    return arr


def run_difference_animation(folder0, folder1, scale_min=None, scale_max=None):
    """
    Load two saved animations, compute option 0 minus option 1, and animate the result.
    """
    json0, npz0 = load_saved_animation(folder0)
    json1, npz1 = load_saved_animation(folder1)

    data0 = np.array(npz0["plotted_variable_array"])
    data1 = np.array(npz1["plotted_variable_array"])

    if data0.shape != data1.shape:
        raise ValueError(
            f"Plotted variable arrays must have the same shape, but got {data0.shape} and {data1.shape}."
        )

    diff = data0 - data1

    if json0["args"][0] != json1["args"][0]:
        print("Warning: the two runs use different shape model files.")
    if json0["kwargs"].get("timesteps_per_day") != json1["kwargs"].get("timesteps_per_day"):
        print("Warning: timesteps_per_day differs between the two runs.")
    if json0["kwargs"].get("solar_distance_au") != json1["kwargs"].get("solar_distance_au"):
        print("Warning: solar_distance_au differs between the two runs.")
    if json0["kwargs"].get("rotation_period_hr") != json1["kwargs"].get("rotation_period_hr"):
        print("Warning: rotation_period_hr differs between the two runs.")

    print("\nRunning difference animation: option 0 minus option 1")
    print(f"Option 0 folder: {folder0}")
    print(f"Option 1 folder: {folder1}")
    print(f"Difference array shape: {diff.shape}")

    # Use the saved animation_frames value to keep the selected timestep stable
    saved_animation_frames = json0["kwargs"].get("animation_frames", 200)

    diff = prepare_difference_array(diff, SELECT_TIMESTEP, saved_animation_frames)
    diff = apply_scale_limits(diff, scale_min, scale_max)
    diff = inject_scale_sentinels(diff, scale_min, scale_max)

    print(f"Final plotted array shape: {diff.shape}")
    if scale_min is not None or scale_max is not None:
        print(f"Using scale limits: min={scale_min}, max={scale_max}")

    args, kwargs, _ = build_animation_args(json0, npz0, diff)

    kwargs["output_dir"] = folder0
    kwargs.setdefault("center_mesh_at_origin", CENTER_MESH_AT_ORIGIN)
    kwargs.setdefault("rotate_mesh", ROTATE_MESH)
    kwargs["colour_map"] = COLOUR_MAP

    animate_model(*args, animation_debug_mode=True, **kwargs)
    print("Animation complete.")


def main():
    script_dir = os.path.abspath(os.path.dirname(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
    base_dir = os.path.join(project_root, "data", "output", "remote_outputs")

    if not os.path.exists(base_dir):
        print(f"Directory {base_dir} does not exist. Exiting.")
        return

    available_folders = get_output_folders(base_dir)
    if not available_folders:
        print(f"No folders found in {base_dir}. Exiting.")
        return

    default_folder0 = available_folders[0]
    folder0_name = get_user_confirmation("option 0", default_folder0, available_folders)

    default_folder1 = available_folders[1] if len(available_folders) > 1 else available_folders[0]
    folder1_name = get_user_confirmation("option 1", default_folder1, available_folders)

    folder0 = os.path.join(base_dir, folder0_name)
    folder1 = os.path.join(base_dir, folder1_name)

    if folder0 == folder1:
        print("Error: option 0 and option 1 must be different folders. Exiting.")
        return

    scale_min = get_optional_float("Enter colour scale minimum, or press Enter for auto: ")
    scale_max = get_optional_float("Enter colour scale maximum, or press Enter for auto: ")

    try:
        run_difference_animation(folder0, folder1, scale_min=scale_min, scale_max=scale_max)
    except Exception as e:
        print(f"Error while running difference animation: {e}")


if __name__ == "__main__":
    main()