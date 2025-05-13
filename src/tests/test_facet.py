import pytest
import numpy as np
from src.model.facet import Facet
from src.model.sub_facet import SubFacet

# Mock Simulation and Config for testing Facet
class MockSimulation:
    def __init__(self):
        self.albedo = 0.1 # Example attribute

class MockConfig:
    def __init__(self, apply_roughness=True, subfacets_count=12, profile_angle=45):
        self.apply_kernel_based_roughness = apply_roughness
        self.roughness_kernel = 'spherical_cap'
        self.kernel_subfacets_count = subfacets_count
        self.kernel_profile_angle_degrees = profile_angle

def test_facet_initialization():
    """Test basic Facet initialization and parent facet functionality."""
    # Create a simple triangular facet
    normal = np.array([0, 0, 1])
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    
    # Create a facet with minimal parameters
    facet = Facet(normal, vertices, timesteps_per_day=100, max_days=1, n_layers=10, calculate_energy_terms=False)
    
    # Test basic attributes
    assert np.allclose(facet.normal, normal)
    assert np.allclose(facet.vertices, vertices)
    assert facet.area == 0.5  # Area of a right triangle with base=1, height=1
    assert np.allclose(facet.position, np.array([1/3, 1/3, 0]))
    
    # Test parent facet attributes
    assert len(facet.sub_facets) == 0
    assert len(facet.parent_incident_energy_packets) == 0
    assert facet.depression_total_absorbed_solar_flux == 0.0

def test_sub_facet_initialization():
    """Test SubFacet initialization as a state-holder."""
    parent_id = 123
    local_id = 5 # This is also the canonical_index
    
    sub_facet = SubFacet(parent_id, local_id)
    
    assert sub_facet.parent_id == parent_id
    assert sub_facet.local_id == local_id
    assert sub_facet.canonical_index == local_id
    assert sub_facet.incident_energy == 0.0
    assert sub_facet.absorbed_energy == 0.0
    assert sub_facet.radiated_energy == 0.0
    # Assert that geometric properties are NOT on SubFacet
    assert not hasattr(sub_facet, 'vertices')
    assert not hasattr(sub_facet, 'normal')
    assert not hasattr(sub_facet, 'area')

def test_generate_spherical_depression_with_canonical_mesh():
    """Test that Facet generates SubFacets using a canonical mesh."""
    normal = np.array([0, 0, 1])
    vertices = np.array([[0,0,0], [1,0,0], [0,1,0]])
    facet = Facet(normal, vertices, 100, 1, 10, False)
    
    # Reset class attribute for clean test environment
    if hasattr(Facet, "_canonical_subfacet_mesh"):
        Facet._canonical_subfacet_mesh = None 
    if hasattr(Facet, "_canonical_mesh_params"):
        delattr(Facet, "_canonical_mesh_params")

    config = MockConfig(apply_roughness=True, subfacets_count=10, profile_angle=30)
    simulation = MockSimulation()
    
    facet.generate_spherical_depression(config, simulation)
    
    assert hasattr(Facet, "_canonical_subfacet_mesh")
    assert Facet._canonical_subfacet_mesh is not None
    assert len(Facet._canonical_subfacet_mesh) > 0 # Assuming your generator makes at least one
    
    expected_subfacet_count = len(Facet._canonical_subfacet_mesh)
    assert len(facet.sub_facets) == expected_subfacet_count
    
    if expected_subfacet_count > 0:
        first_sub_facet = facet.sub_facets[0]
        assert first_sub_facet.parent_id == id(facet)
        assert first_sub_facet.local_id == 0
        assert first_sub_facet.canonical_index == 0

        # Check that canonical mesh has expected keys (vertices, normal, area)
        canonical_entry = Facet._canonical_subfacet_mesh[0]
        assert 'vertices' in canonical_entry
        assert 'normal' in canonical_entry
        assert 'area' in canonical_entry

    # Test that calling it again with same config doesn't regenerate (by checking object ID, or add counter)
    mesh_id_before = id(Facet._canonical_subfacet_mesh)
    facet.generate_spherical_depression(config, simulation)
    mesh_id_after = id(Facet._canonical_subfacet_mesh)
    assert mesh_id_before == mesh_id_after

    # Test regeneration if config changes
    config_new = MockConfig(apply_roughness=True, subfacets_count=20, profile_angle=60)
    facet.generate_spherical_depression(config_new, simulation)
    mesh_id_new_config = id(Facet._canonical_subfacet_mesh)
    assert mesh_id_after != mesh_id_new_config # Should be a new mesh object
    assert len(facet.sub_facets) == len(Facet._canonical_subfacet_mesh)


if __name__ == '__main__':
    pytest.main([__file__]) 