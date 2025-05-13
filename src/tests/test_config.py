import os
import tempfile
import yaml
import pytest
from src.utilities.config import Config

def test_spherical_depression_roughness_config():
    """Test that kernel-based roughness parameters are loaded correctly."""
    # Create a temporary config file with custom values
    test_config_data = {
        'shape_model_file': 'test.stl',  # Required field
        'apply_kernel_based_roughness': True,
        'roughness_kernel': 'spherical_cap',
        'kernel_subfacets_count': 50,
        'kernel_profile_angle_degrees': 30,
        'kernel_directional_bins': 72
    }
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config_data, f)
        temp_config_path = f.name
    
    try:
        # Load the config
        config = Config(temp_config_path)
        
        # Test the values were loaded correctly
        assert config.apply_kernel_based_roughness is True
        assert config.roughness_kernel == 'spherical_cap'
        assert config.kernel_subfacets_count == 50
        assert config.kernel_profile_angle_degrees == 30
        assert config.kernel_directional_bins == 72
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_config_path)

if __name__ == '__main__':
    pytest.main([__file__]) 