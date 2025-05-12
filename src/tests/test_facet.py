import pytest
import numpy as np
from src.model.facet import Facet
from src.model.sub_facet import SubFacet
from src.utilities.config import Config

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
    """Test SubFacet initialization."""
    parent_id = 123
    local_id = 0
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    
    sub_facet = SubFacet(parent_id, local_id, vertices)
    
    assert sub_facet.parent_id == parent_id
    assert sub_facet.local_id == local_id
    assert np.allclose(sub_facet.vertices, vertices)
    assert np.allclose(sub_facet.normal, np.array([0, 0, 1]))
    assert sub_facet.area == 0.5
    assert np.allclose(sub_facet.position, np.array([1/3, 1/3, 0]))
    assert sub_facet.incident_energy == 0.0
    assert sub_facet.absorbed_energy == 0.0
    assert sub_facet.radiated_energy == 0.0

if __name__ == '__main__':
    pytest.main([__file__]) 