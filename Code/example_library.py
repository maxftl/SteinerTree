from TriGraph import *
from SubmodularFunction import *

def ex_shortest_path():
    example = {}
    V = np.array([ [0,0],[1,0],[0,1],[1,1] ])
    F = np.array([ [0,1,2],[1,2,3] ], dtype = np.int)
    example['graph'] = TriGraph(V,F)
    objective_data = {
        (0,): 0,
        (1,): 1
    }
    example['objective'] = SubmodularFunction(objective_data, 1)
    example['supply'] = np.array( [ [1],[0],[0],[-1] ] )
    example['dim'] = 1
    return example

def ex_steiner_tree():
    example = {}
    V = np.array([ [0,1],[0,-1],[3,0] ])
    F = np.array([ [0,1,2] ], dtype = np.int)
    example['graph'] = TriGraph(V,F)
    objective_data = {
        (0,0): 0,
        (1,0): 1,
        (0,1): 1,
        (1,1): 1
    }
    example['objective'] = SubmodularFunction(objective_data, 2)
    example['supply'] = np.array( [ [1,0],[0,1],[-1,-1] ] )
    example['dim'] = 2
    return example

def ex_shortest_path_2d():
    example = {}
    V = np.array([ [0,1],[0,-1],[3,0] ])
    F = np.array([ [0,1,2] ], dtype = np.int)
    example['graph'] = TriGraph(V,F)
    objective_data = {
        (0,0): 0,
        (1,0): 1,
        (0,1): 1,
        (1,1): 2
    }
    example['objective'] = SubmodularFunction(objective_data, 2)
    example['supply'] = np.array( [ [1,0],[0,1],[-1,-1] ] )
    example['dim'] = 2
    return example