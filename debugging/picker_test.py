import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

class STLViewer:
    def __init__(self, stl_file):
        # Load STL file
        self.mesh = mesh.Mesh.from_file(stl_file)
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Prepare vertices and plot triangles
        self.vertices = self.mesh.vectors
        self.centroids = np.mean(self.vertices, axis=1)
        self.collection = Poly3DCollection(self.vertices, alpha=0.7, linewidths=1, edgecolors='k', facecolors='grey')
        self.ax.add_collection3d(self.collection)

        # Set the scale of the axes
        self.ax.auto_scale_xyz(self.vertices[:, :, 0], self.vertices[:, :, 1], self.vertices[:, :, 2])

        # Connect the event for picking facets
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)
        self.collection.set_picker(True)

    def on_pick(self, event):
        if event.artist != self.collection:
            return

        # Use the camera's transformation matrix to calculate depth for centroids
        m = self.ax.get_proj()
        picked_indices = event.ind
        # Correctly create homogeneous coordinates for the centroids
        centroids_homogeneous = np.hstack((self.centroids[picked_indices], np.ones((len(picked_indices), 1))))
        # Transform centroids using the projection matrix and calculate depths from the transformed z-coordinates
        transformed_centroids = (m @ centroids_homogeneous.T).T
        depths = transformed_centroids[:, 2]
        frontmost_index = picked_indices[np.argmin(depths)]  # The closest centroid in z from the camera's perspective

        # Only keep the frontmost facet colored, and reset others
        face_colors = np.tile([0.5, 0.5, 0.5, 0.7], (len(self.vertices), 1))
        face_colors[frontmost_index] = [1, 0, 0, 1]  # Set new facet to red
        self.collection.set_facecolors(face_colors)
        self.fig.canvas.draw()

    def show(self):
        plt.show()

# Usage
viewer = STLViewer('500m_cube.stl')
viewer.show()
