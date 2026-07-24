import numpy as np
from scipy.spatial import cKDTree

radius = 3390 #km

class coordMap:
    def __make_tree_(self, long, lat):
        long = long.ravel()
        lat = lat.ravel()
        points = np.column_stack((long, lat))
        tree = cKDTree(points)
        return tree

