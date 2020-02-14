import numpy as np
from itertools import permutations

class SubmodularFunction:

    def __init__(self, data, dim): #data: dictionary of values at {0,1}-points
        self.data = data
        self.dim = dim
    #Chain: permutation of indices 0...dim-1, subgradient is taken wrt {chain_0}, {chain_0, chain_1}...
    def subgradient(self, chain):
        start = [0]*self.dim
        last = self.data[tuple(start)]
        result = np.zeros(self.dim)
        for c in chain:
            start[c] = 1
            curr = self.data[tuple(start)]
            result[c] = curr - last
            last = curr
        return result

    #vertices of subdifferential at 0
    def subdifferential0(self):
        result = []
        for chain in permutations(range(self.dim)):
            result.append(self.subgradient(chain))
        return np.array(result)

    '''Return (a,b) such that x \in dh(0) <=> ax <= b for all (a,b)'''
    def subdifferential0_inequalities(self):
        subset_masks = [ tuple((x>>i)&1 for i in range(self.dim)) for x in range(1 << self.dim) ]
        inequalities = [ (np.array(s),self.data[s]) for s in subset_masks ]
        #inequalities += [ (-np.array(s),-self.data[s]) for s in subset_masks ]
        return inequalities


