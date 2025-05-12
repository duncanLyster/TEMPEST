import os
import tempfile
import yaml
import pytest
from src.utilities.config import Config

def test_spherical_depression_roughness_config():
    """Test that spherical depression roughness parameters are loaded correctly."""
    # Create a temporary config file with custom values
    test_config_data = {
        'shape_model_file': 'test.stl',  # Required field
        'apply_spherical_depression_roughness': True,
        'depression_subfacets_count': 50,
        'depression_profile_angle_degrees': 30,
        'depression_MCRT_rays_per_emission_step': 200,
        'depression_internal_scattering_iterations': 3,
        'depression_outgoing_emission_bins': 72
    }
    
    # Write to a temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(test_config_data, f)
        temp_config_path = f.name
    
    try:
        # Load the config
        config = Config(temp_config_path)
        
        # Test the values were loaded correctly
        assert config.apply_spherical_depression_roughness is True
        assert config.depression_subfacets_count == 50
        assert config.depression_profile_angle_degrees == 30
        assert config.depression_MCRT_rays_per_emission_step == 200
        assert config.depression_internal_scattering_iterations == 3
        assert config.depression_outgoing_emission_bins == 72
        
    finally:
        # Clean up the temporary file
        os.unlink(temp_config_path)

if __name__ == '__main__':
    pytest.main([__file__]) 