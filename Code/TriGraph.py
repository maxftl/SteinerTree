import numpy as np

'''
Edges are ordered lexicographically wrt to the vertices e=(v,w)
'''
class TriGraph:
    def __init__(self, V, F):
        self.num_vertices = np.size(V,0)
        self.num_faces = np.size(F, 0)
        self.V = V
        self.F = F
        self.adjacency = np.zeros( (self.num_vertices, self.num_vertices), dtype=np.int8 ) #Maybe make sparse later
        self.edge_lengths = np.zeros( (self.num_vertices,self.num_vertices) ) #(i,j) th entry contains length of the corresponding edge, or -1 if there is none
        self._init_adjacency()
        self._init_edges()

    def _init_adjacency(self):
        for face in self.F:
            for i in range(3):
                f1 = face[i]
                f2 = face[(i+1)%3]
                self.adjacency[f1][f2] = np.sign(f2 - f1)
                self.adjacency[f2][f1] = -np.sign(f2 - f1)
                length = np.linalg.norm(self.V[f1,:]-self.V[f2,:])
                self.edge_lengths[f1][f2] = length
                self.edge_lengths[f2][f1] = length

    def _init_edges(self):
        self.edges = set()
        for face in self.F:
            for i in range(3):
                f1 = face[i]
                f2 = face[(i+1)%3]
                self.edges.add( tuple(sorted( (f1,f2) ) ) )
        self.edges = sorted(list(self.edges))
        self.edge_to_index = dict( (self.edges[i],i) for i in range(len(self.edges)) )
        self.num_edges = len(self.edges)
        self.vertex_to_edges = dict( (i,[e for e in range(self.num_edges) if self.edges[e][0] == i or self.edges[e][1] == i]) for i in range(self.num_vertices) )
        


    def save(self, filename):
        np.savez(filename,V=self.V,F=self.F)

    @staticmethod
    def load(filename):
        npzfile = np.load(filename)
        return TriGraph(npzfile['V'], npzfile['F'])

    ''' vertex_fun d x num_vertices matrix; vertex_fun[:,i] value at vertex i '''
    def element_wise_differential(self, vertex_fun):
        face_trafos = [  (self.V[(v[1],v[2]),:]-self.V[(v[0],v[0]),:]).T for v in self.F ]
        tr_diff = [ (vertex_fun[:,(v[1],v[2])]-vertex_fun[:,(v[0],v[0])]) for v in self.F ]
        result = [ np.dot(Phi,np.linalg.inv(T)) for (Phi,T) in zip(tr_diff,face_trafos)]
        return result

