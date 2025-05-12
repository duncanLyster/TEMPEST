import numpy as np

class SubFacet:
    def __init__(self, parent_id, local_id, vertices, normal=None):
        """
        Initialize a SubFacet object.
        
        Args:
            parent_id: ID of the parent facet this sub-facet belongs to
            local_id: Local ID within the parent's depression
            vertices: 3x3 numpy array of vertex coordinates
            normal: Optional normal vector (will be calculated if not provided)
        """
        self.parent_id = parent_id
        self.local_id = local_id
        self.vertices = np.array(vertices, dtype=np.float64)
        
        # Calculate geometric properties
        if normal is None:
            self.normal = self._calculate_normal()
        else:
            self.normal = np.array(normal, dtype=np.float64)
            self.normal /= np.linalg.norm(self.normal)
            
        self.area = self._calculate_area()
        self.position = np.mean(self.vertices, axis=0)
        
        # Energy tracking (will be used in later steps)
        self.incident_energy = 0.0  # Energy received in current timestep
        self.absorbed_energy = 0.0  # Energy absorbed in current timestep
        self.radiated_energy = 0.0  # Energy available for radiation in current timestep
        
    def _calculate_normal(self):
        """Calculate the normal vector of the sub-facet."""
        v1, v2, v3 = self.vertices
        normal = np.cross(v2 - v1, v3 - v1)
        return normal / np.linalg.norm(normal)
    
    def _calculate_area(self):
        """Calculate the area of the sub-facet."""
        v1, v2, v3 = self.vertices
        return np.linalg.norm(np.cross(v2 - v1, v3 - v1)) / 2 