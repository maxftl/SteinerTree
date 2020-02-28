import numpy as np
from TopoGraph import *
from TriGraph import *
from scipy.spatial import Delaunay

def triangulate(topo):
    tri = Delaunay(topo.positions)
    return TriGraph(topo.positions,tri.simplices)