from TriGraph import *
from SubmodularFunction import *

def ex_shortest_path(num_blocks_x, num_blocks_y):
    assert(num_blocks_x > 0 and num_blocks_y > 0)
    example = {}
    V = np.array([ (x,y) for y in range(num_blocks_y+1) for x in range(num_blocks_x+1) ])
    F1 = np.array([ (x+y*(num_blocks_x+1),(num_blocks_x+1)*y+x+1,x+(y+1)*(num_blocks_x+1) ) for y in range(num_blocks_y) for x in range(num_blocks_x) ], dtype = np.int)
    F2 = np.array([ (x+(y+1)*(num_blocks_x+1),(num_blocks_x+1)*(y+1)+x+1,x+y*(num_blocks_x+1)+1 ) for y in range(num_blocks_y) for x in range(num_blocks_x) ], dtype = np.int)
    F = np.vstack( (F1,F2) )
    example['graph'] = TriGraph(V,F)
    objective_data = {
        (0,): 0,
        (1,): 1
    }
    example['objective'] = SubmodularFunction(objective_data, 1)
    example['supply'] = np.array( [[1]] + [ [0] for _ in range( (num_blocks_x+1)*(num_blocks_y+1)-2 ) ] + [[-1]] )
    example['dim'] = 1
    return example


def ex_simple_2d(q):
    example = {}
    V = np.array([ [0,1],[0,-1],[3,0] ])
    F = np.array([ [0,1,2] ], dtype = np.int)
    example['graph'] = TriGraph(V,F)
    objective_data = {
        (0,0): 0,
        (1,0): 1,
        (0,1): 1,
        (1,1): 1+q
    }
    example['objective'] = SubmodularFunction(objective_data, 2)
    example['supply'] = np.array( [ [1,0],[0,1],[-1,-1] ] )
    example['dim'] = 2
    return example

def ex_simple_3d(q,r):
    example = {}
    V = np.array( [ [0,1],[0,0],[0,-1],[3,0] ] )
    F = np.array([ [0,1,3],[1,2,3] ], dtype = np.int)
    example['graph'] = TriGraph(V,F)
    objective_data = {
        (0,0,0): 0,
        (1,0,0): 1,
        (0,1,0): 1,
        (0,0,1): 1,
        (1,1,0): 1+q,
        (1,0,1): 1+q,
        (0,1,1): 1+q,
        (1,1,1): 1+q+r
    }
    example['objective'] = SubmodularFunction(objective_data, 3)
    example['supply'] = np.array( [ [1,0,0],[0,1,0],[0,0,1],[-1,-1,-1] ] )
    example['dim'] = 3
    return example

def ex_steiner_tree(n_sources):
    eps = .1
    example = {}
    V = np.vstack((np.hstack( (np.zeros((n_sources,1)),(np.linspace(-1,1,n_sources)+eps*np.random.rand(n_sources)).reshape((n_sources,1)) ) ),[3,0]))
    F = np.array( [ [i,i+1,n_sources] for i in range(n_sources-1) ] ,dtype = np.int)
    objective_data = dict([ ( tuple( (x>>i)&1 for i in range(n_sources)), int(x!=0) ) for x in range(1<<n_sources) ])
    example['graph'] = TriGraph(V,F)
    example['objective'] = SubmodularFunction(objective_data, n_sources)
    example['dim'] = n_sources
    example['supply'] = np.vstack( (np.eye(n_sources),-np.ones(n_sources)) )
    return example