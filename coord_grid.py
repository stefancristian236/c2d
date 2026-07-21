import numpy as np
from scipy.spatial import cKDTree

class CoordMap:
    def __init__(self, latitude, longitude, data):
        self.shape = latitude.shape 
        self.data = data
        
        self.lat_flat = latitude.ravel()
        self.lon_flat = longitude.ravel()
        self.data_flat = data.ravel()
        
        self.points = np.column_stack(
            [
                self.lat_flat, 
                self.lon_flat
            ]
        )
        self.tree = cKDTree(self.points)

    def nearest(self, lat, lon):
        
        dist, idx = self.tree.query(
            [
                lat, 
                lon
            ]
        )
        
        row, col = np.unravel_index(idx, self.shape)
        return self.data_flat[idx], (row, col), dist
    
    def nearest_batch(self, lats, lons):
        pts = np.column_stack(
            [
                lats,
                lons
            ]
        )
        dist, idx = self.tree.query(
            pts
        )
        rows, cols = np.unravel_index(
            idx,
            self.shape
        )
        
        return self.data_flat[idx], rows, cols, dist

    def region(self, lat_min, lat_max, lon_min, lon_max):
        mask = (
            (self.lat_flat >= lat_min) & (self.lat_flat <= lat_max) &
            (self.lon_flat >= lon_min) & (self.lon_flat <= lon_max)
        )
        return self.data_flat[mask], self.lat_flat[mask], self.lon_flat[mask]

    def patch(self, lat, lon):
        _, (row, col), _ = self.nearest(lat, lon)
        r0, r1 = max(0, row), min(self.shape[0], row + 1)
        c0, c1 = max(0, col), min(self.shape[1], col + 1)
        
        return self.data[r0 : r1, c0 : c1]