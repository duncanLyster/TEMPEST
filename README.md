![TEMPEST Banner](resources/documentation/banner.jpg)

# Thermophysical Body Model Simulation Script
This Python script simulates diurnal temperature variations of a solar system body based on a given shape model. Initially designed for ESAs Comet Interceptor Mission planning as part of my research at the University of Oxford, it's adaptable for asteroids and other planetary bodies like Enceladus' surface fractures. As of 14 November 2024 it is stable and a range of test shape models converge well, but it is still under development to improve ease of use, computation speed, and validity in a wider range of parameter spaces. 

## Features
- Simulates temperature variations considering material and model properties.
- Calculates insolation and temperature arrays, iterating until model convergence.
- Visualises results and saves for further analysis.
- SI units are standard, exceptions are clearly stated.

## Requirements
Ensure the following Python packages are installed:

- numpy
- matplotlib
- pandas
- numba
- joblib
- numpy-stl
- tqdm
- scipy
- seaborn
- scikit-learn
- pyvista
- vtk
- pyyaml

You can install these dependencies using:
`pip install -r requirements.txt`

Additionally, ensure you have:
- Python 3.x installed
- An STL file of the body shape in ASCII format

## Usage
1. Ensure all dependencies are installed.
2. Update settings in `config.yaml`
3. Run the script with Python: `python thermophysical_body_model.py`

## Known issues and limitations
Please carefully read header notes in thermophysical_body_model.py for known bugs and limitations.

## Getting Started
The script workflow involves:

1. Reading the provided shape model.
2. Setting up material and model parameters.
3. Calculating insolation and temperature arrays.
4. Iterating through the model until convergence.
5. Saving and visualising results for further analysis.

Tip: If your shape model is not in ASCII .stl format, you can use Blender to convert it.

## Contribution
Feel free to fork the project for custom enhancements or issue tracking on GitHub: https://github.com/duncanLyster/comet_nucleus_model

## Author and Acknowledgements
Duncan Lyster | Started: 15 Feb 2024

With contributions from: Joe Penn, Maisie Rashman, and Bea Chikani

## Licence
This project is open-source and available under the MIT License - see the License.md file for details.


