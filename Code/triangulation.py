import numpy as np
from TopoGraph import *
from TriGraph import *
import triangle as tr

def triangulate(topo):
    tri = {'vertices':topo.positions, 'segments': topo.edges}
    triangulation = tr.triangulate(tri)
    return TriGraph(triangulation['vertices'],triangulation['triangles'])





