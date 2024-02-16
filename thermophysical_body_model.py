''' 
This Python script simulates diurnal temperature variations of a solar system body based on
a given shape model. It reads in the shape model, sets material and model properties, calculates 
insolation and temperature arrays, and iterates until the model converges. The results are saved and 
visualized.

It was built as a tool for planning the comet interceptor mission, but is intended to be 
general enough to be used for asteroids, and other planetary bodies e.g. fractures on 
Enceladus' surface.

All calculation figures are in SI units, except where clearly stated otherwise.

Full documentation to be found (one day) at: **INSERT LINK TO GITHUB REPOSITORY**.

OPEN QUESTIONS: 
Do we consider partial shadow? 
Do we treat facets as points or full 2D polygons?
Should insolation be calculated once for each facet as a 2D map where the sun later passes over? Or just as a curve each time the model runs? 

EXTENSIONS: 
Binaries: Complex shading from non-rigid geometry (Could be a paper) 
Add temporary heat sources. 

IMPORTANT CONSIDERATIONS: 
Generalising the model so it can be used e.g for asteroids, Enceladus fractures, adjacent emitting bodies (e.g. binaries, Saturn) 

'''

# Import any necessary libraries

# Define global variables
# Comet-wide material properties (currently placeholders)
emmisivity = 0.5                    # Dimensionless
albedo = 0.5                        # Dimensionless
thermal_conductivity = 1.0          # W/mK 
density = 1.0                       # kg/m^3
specific_heat_capacity = 1.0        # J/kgK
beaming_factor = 0.5                # Dimensionless

# Model setup parameters
layer_thickness = 0.1               # m (this may be calculated properly from insolation curve later, but just a value for now)
n_layers = 10                       # Number of layers in the conduction model
solar_distance = 1.0                # AU
solar_constant = 1361               # W/m^2 Solar constant is the solar radiation received per unit area at 1 AU (could use luminosity of the sun and distance to the sun to calculate this)
time_step = 1000                    # s
rotation_period = 100,000           # s (1 day on the comet)
max_days = 5                        # Maximum number of days to run the model for
rotation_axis = [0, 0, 1]           # Unit vector pointing along the rotation axis
body_orientation = [0, 0, 1]        # Unit vector pointing along the body's orientation


# Define any necessary functions
def read_shape_model():
    ''' 
    This function reads in the shape model of the comet. It is currently a placeholder, and will be replaced with a proper shape model file reader.
    '''
    facet_array = "placeholder" # The facet array will 
    return facet_array

def visualise_shape_model():
    ''' 
    This function visualises the shape model of the comet to allow the user to intuiutively check the setup is as intended. It is currently a placeholder, but when complete it will show an animation of the comet rotating relative to the sun from an external observers position. Rotation axis, and period will be shown.
    '''
    pass

def initialise_data_cube():
    ''' 
    This function sets up the data cube, which will store the geometry, temperatures, insolation arrays (not sure if 1D or 2D) and secondary radiation arrays for each layer of the model. It is currently a placeholder, and will be replaced with a proper data cube initialiser.


    Facet#	Area	Position	Normal vector 	T0_t0	T0_t1	T1_t0 	T1_t1 etc	Insolation curve	Secondary radiation 
    1	    Float	Coordinates	Vector (3D)	    Float	Float	Float	Float 	    1D array (or 2D?) 	2D array [index, geometric coefficient]

    QUESTIONS:
    1) Should the insolation curve be a 1D array or a 2D array?
    2) Do we save every temperature for every time step, or just one per day plus the most recent?

    '''
    data_cube = "placeholder"
    							
    return data_cube

def main():
    ''' 
    This is the main program for the thermophysical body model. It calls the necessary functions to read in the shape model, set the material and model properties, calculate insolation and temperature arrays, and iterate until the model converges. The results are saved and visualized.
    '''
    # Get the shape model
    shape_model = read_shape_model()
    print(shape_model)

    # Visualise the shape model
    visualise_shape_model()

    # Setup the data cube
    data_cube = initialise_data_cube()

    # Calculate insolation and temperature arrays

    # Calulate initial temperature array

    # Calculate secondary radiation array
        # Ray tracing to work out which facets are visible from each facet
        # Calculate the geometric coefficient of secondary radiation from each facet
        # Write the index and coefficient to the data cube
    
    # while (not converged) and (days < max_days):
        #for time in range(0, rotation_period, time_step):
            # Calculate new temperatures everywhere in the model
            #for facet in shape_model:
                # Calculate insolation
                # Calculate secondary radiation
                # Calculate conducted heat
                # Calculate re-emitted radiation
                # Calculate sublimation energy loss
                # Put the above into equation for new surface temperature
                # Calculate new temperatures for all sub-surface layers
                # Save the new temperatures to the data cube
                # Set T = T_new
            
            # Calculate convergence factor

    # Save the data cube

    # Visualise the results - animation of 1 day 

    # Output the results to a file that can be used with ephemeris to produce instrument simulations

# Call the main program to start execution
if __name__ == "__main__":
    main()
