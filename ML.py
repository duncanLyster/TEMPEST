import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Define a function to calculate comet temperature given input parameters
def calculate_comet_temperature(input_parameters):

    # Assign input parameters
    emissivity, albedo, thermal_conductivity, density, specific_heat_capacity, beaming_factor, \
    solar_distance_au, solar_luminosity, = input_parameters

    # Load the final day temperatures
    final_day_temperatures = np.loadtxt('outputs/final_day_temperatures.csv', delimiter=',')

    # Return the mean temperature of the comet
    return np.mean(final_day_temperatures)

# Define a function to perform linear regression
def perform_linear_regression(X, y, parameter_name):
    # Reshape X and y
    X = X.reshape(-1, 1)
    y = y.reshape(-1, 1)

    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Predict y values
    y_pred = model.predict(X)

    # Plot the results
    plt.scatter(X, y, color='blue', label='Actual data')
    plt.plot(X, y_pred, color='red', label='Linear regression')
    plt.xlabel(parameter_name)
    plt.ylabel('Comet Temperature (K)')
    plt.title(f'Linear Regression: {parameter_name} vs Comet Temperature')
    plt.legend()
    plt.show()

# Read input parameters from CSV file
input_parameters_df = pd.read_csv('Datasheet_comet.csv')

input_parameters_df = input_parameters_df.dropna()


# Loop through each row in the dataframe
for index, row in input_parameters_df.iterrows():
    # Convert row to list (input parameters)
    input_parameters = row.tolist()

    # Calculate comet temperature
    comet_temperature = calculate_comet_temperature(input_parameters)

    # Append comet temperature to the dataframe
    input_parameters_df.at[index, 'Comet Temperature'] = comet_temperature

# Perform linear regression for each input parameter against comet temperature
for column in input_parameters_df.columns[1:]:
    X = input_parameters_df[column].values
    y = input_parameters_df['Comet Temperature'].values
    perform_linear_regression(X, y, column)

