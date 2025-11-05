"""
Combine facet data CSV files with interpolation to 24 hours at 0.1 hour intervals.

This script reads insolation and temperature data for multiple facets and thermal inertia
values from the facet_data folder, interpolates them to a uniform 0.1 hour grid, and 
combines them into a single CSV file.

Author: Duncan Lyster
Created: 2024
"""

import os
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

def combine_facet_data(input_folder='facet_data', output_file='facet_data/combined_facet_data_interpolated.csv'):
    """
    Combine insolation and temperature CSV files with interpolation.
    
    Args:
        input_folder (str): Path to folder containing the CSV files
        output_file (str): Path for the output combined CSV file
    """
    
    print("=" * 60)
    print("Combining Facet Data with Interpolation")
    print("=" * 60)
    
    # Define the target time grid: 0 to 24 hours at 0.1 hour intervals
    target_time_hrs = np.arange(0, 24, 0.1)
    n_points = len(target_time_hrs)
    print(f"\nTarget grid: {n_points} points from 0 to {target_time_hrs[-1]:.1f} hours")
    print(f"Time interval: 0.1 hours (6 minutes)")
    
    # Initialize the combined dataframe with the time column
    combined_data = {'Local_Time_hrs': target_time_hrs}
    
    # Read insolation data
    insolation_file = os.path.join(input_folder, 'insolation_local_time.csv')
    if not os.path.exists(insolation_file):
        raise FileNotFoundError(f"Insolation file not found: {insolation_file}")
    
    print(f"\nReading insolation data from: {insolation_file}")
    df_insolation = pd.read_csv(insolation_file)
    
    # Convert degrees to hours (0-360 degrees -> 0-24 hours)
    original_time_hrs = df_insolation['Local_Time_deg'].values / 360.0 * 24.0
    
    # Get facet columns (all except the time column)
    facet_columns = [col for col in df_insolation.columns if col.startswith('Facet_')]
    facets = [col.split('_')[1] for col in facet_columns]
    
    print(f"Found {len(facets)} facets: {', '.join(facets)}")
    
    # Find all temperature files with different TI values
    temp_files = sorted(Path(input_folder).glob('temperature_local_time_TI_*.csv'))
    
    if not temp_files:
        raise FileNotFoundError(f"No temperature files found in {input_folder}")
    
    print(f"\nFound {len(temp_files)} temperature files")
    
    # Extract TI values from filenames
    ti_values = [temp_file.stem.split('_TI_')[1] for temp_file in temp_files]
    print(f"TI values: {', '.join(ti_values)}")
    
    # Load all temperature data into a dictionary
    temp_data = {}
    for temp_file in temp_files:
        ti_value = temp_file.stem.split('_TI_')[1]
        df_temp = pd.read_csv(temp_file)
        temp_time_hrs = df_temp['Local_Time_deg'].values / 360.0 * 24.0
        temp_data[ti_value] = {
            'time': temp_time_hrs,
            'data': df_temp
        }
    
    # Process each facet: add insolation followed by all temperature columns
    for facet_col in facet_columns:
        facet_id = facet_col.split('_')[1]
        print(f"\n  Processing Facet {facet_id}:")
        
        # Add insolation for this facet
        original_data = df_insolation[facet_col].values
        interp_func = interp1d(original_time_hrs, original_data, kind='linear', 
                              bounds_error=False, fill_value='extrapolate')
        interpolated_data = interp_func(target_time_hrs)
        column_name = f'Insolation_Facet_{facet_id}'
        combined_data[column_name] = interpolated_data
        print(f"    Added: {column_name}")
        
        # Add temperature data for all TI values for this facet
        for ti_value in ti_values:
            df_temp = temp_data[ti_value]['data']
            temp_time_hrs = temp_data[ti_value]['time']
            
            original_data = df_temp[facet_col].values
            interp_func = interp1d(temp_time_hrs, original_data, kind='linear',
                                  bounds_error=False, fill_value='extrapolate')
            interpolated_data = interp_func(target_time_hrs)
            column_name = f'Temp_Facet_{facet_id}_TI_{ti_value}'
            combined_data[column_name] = interpolated_data
            print(f"    Added: {column_name}")
    
    # Create DataFrame and save to CSV
    df_combined = pd.DataFrame(combined_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save to CSV
    df_combined.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print(f"SUCCESS! Combined data saved to: {output_file}")
    print("=" * 60)
    print(f"\nOutput file statistics:")
    print(f"  Number of rows: {len(df_combined)}")
    print(f"  Number of columns: {len(df_combined.columns)}")
    print(f"  Time range: {df_combined['Local_Time_hrs'].min():.1f} to {df_combined['Local_Time_hrs'].max():.1f} hours")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.1f} KB")
    
    print(f"\nColumn names:")
    for i, col in enumerate(df_combined.columns, 1):
        print(f"  {i:2d}. {col}")
    
    return df_combined


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Combine facet data CSV files with interpolation to 0.1 hour intervals',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default paths (facet_data folder)
  python scripts/combine_facet_data.py
  
  # Specify custom input folder
  python scripts/combine_facet_data.py --input my_data_folder
  
  # Specify custom output file
  python scripts/combine_facet_data.py --output results/combined.csv
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='facet_data',
        help='Input folder containing CSV files (default: facet_data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='facet_data/combined_facet_data_interpolated.csv',
        help='Output file path (default: facet_data/combined_facet_data_interpolated.csv)'
    )
    
    args = parser.parse_args()
    
    try:
        combine_facet_data(input_folder=args.input, output_file=args.output)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

