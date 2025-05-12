import numpy as np

class SubFacet:
    def __init__(self, parent_id, local_id, vertices=None, normal=None):
        """
        Initialize a SubFacet object.
        
        Args:
            parent_id: ID of the parent facet this sub-facet belongs to
            local_id: Local ID within the parent's depression (can also be the canonical_index)
        """
        self.parent_id = parent_id
        self.local_id = local_id # This will also serve as the index into the canonical mesh list
        self.canonical_index = local_id # Explicitly store the index to the canonical mesh entry
        
        # Geometric properties are no longer stored directly
        # self.vertices = np.array(vertices, dtype=np.float64) 
        # self.normal = ...
        # self.area = ...
        # self.position = ...
        
        # Energy tracking (will be used in later steps)
        self.incident_energy = 0.0  # Energy received in current timestep
        self.absorbed_energy = 0.0  # Energy absorbed in current timestep
        self.radiated_energy = 0.0  # Energy available for radiation in current timestep
        # Add temperature arrays and other thermal state variables here as needed in future steps
        
    # Remove geometric calculation methods
    # def _calculate_normal(self): ...
    # def _calculate_area(self): ... 