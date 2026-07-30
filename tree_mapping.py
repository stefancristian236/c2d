import numpy as np

class HyperSpectralTree:
    def __init__(self):
        self.parent = None
        self.children = None
        
    def parent(self, parent):
        self.parent = parent
        
    def children(self, values):
        self.children = values