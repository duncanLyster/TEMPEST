import os

def convert_stl_m_to_km(input_file, output_file=None):
    """
    Convert an ASCII STL file from meters to kilometers.
    
    Args:
        input_file (str): Path to the input STL file
        output_file (str, optional): Path to the output file. If None, uses input_file with '_km' suffix
    """
    if output_file is None:
        # Create output filename by adding '_km' before the extension
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_km{ext}"
    
    print(f"Converting {input_file} to kilometers...")
    print(f"Output will be saved to {output_file}")
    
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.strip().startswith('vertex'):
                # Split the line into parts
                parts = line.strip().split()
                if len(parts) == 4:  # "vertex x y z"
                    # Convert the coordinates from m to km
                    x = float(parts[1]) / 1000.0
                    y = float(parts[2]) / 1000.0
                    z = float(parts[3]) / 1000.0
                    
                    # Reconstruct the line with the same spacing as before
                    indent = line[:line.find('vertex')]
                    new_line = f"{indent}vertex {x:.6f} {y:.6f} {z:.6f}\n"
                    f_out.write(new_line)
                else:
                    f_out.write(line)  # If not properly formatted, keep as is
            else:
                f_out.write(line)  # Keep non-vertex lines unchanged
    
    print("Conversion complete!")
    return output_file

if __name__ == "__main__":
    # Use the specified file name
    input_file = "DJ_v07_Duncan_Lyster.stl"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found!")
    else:
        # Create output filename
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_km{ext}"
        
        # Convert the file
        convert_stl_m_to_km(input_file, output_file)