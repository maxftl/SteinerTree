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
        self._init_adjacency()
        self._init_edges()

    def _init_adjacency(self):
        self.adjacency = np.zeros( (self.num_vertices, self.num_vertices), dtype=np.int8 ) #Maybe make sparse later
        self.edge_lengths = np.zeros( (self.num_vertices,self.num_vertices) ) #(i,j) th entry contains length of the corresponding edge, or -1 if there is none
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

    ''' returns index in face with face_id such that face[i] is incident vertex'''
    def _find_incident_vertex(self, face_id, edge_direction):
        face = self.F[face_id,:]
        vertices = self.V[face,:].T
        for i in range(3):
            j, k = ( (i+1+r)%3 for r in range(2) )
            fi, fj, fk = (i,j,k)
            A = vertices[:,(fj,fk)] - vertices[:,(fi,fi)]
            x = np.linalg.solve(A,edge_direction)
            if x[0]*x[1] > 0:
                return i

    def _clear_hanging_vertex(self, vertex_id, old_edge):
        fi,fj = old_edge
        h_faces = [(i,f) for i,f in enumerate(self.F) if fi in f and fj in f]
        if len(h_faces) > 1:
            raise Exception('Too many hanging faces')
        if len(h_faces) == 0:
            #print('no edges to add')
            return
        hf = h_faces[0][1]
        hf_id = h_faces[0][0]
        self.F = np.vstack( (self.F,hf) )
        fk = [f_ for f_ in hf if f_ != fi and f_ != fj][0]
        self.F[hf_id,:] = [fi,vertex_id,fk]
        self.F[-1,:] = [fj,vertex_id,fk]
        self.num_faces += 1
        

    def add_edge_to_face(self, face_id, edge_direction):
        face = self.F[face_id,:]
        vertices = self.V[face,:]
        i = self._find_incident_vertex(face_id, edge_direction)
        j, k = ( (i+1+r)%3 for r in range(2) )
        fi, fj, fk = (face[i],face[j],face[k])
        other_edge = vertices[j,:]-vertices[k,:]
        A = np.array( [edge_direction,other_edge] ).T
        x = np.linalg.solve(A,vertices[j,:]-vertices[i,:])
        new_position = vertices[i,:] + x[0] * edge_direction
        new_index = self.V.shape[0]
        self.V = np.vstack( (self.V, new_position) )
        self.F = np.vstack( (self.F, face) )
        self.F[face_id,k] = new_index
        self.F[-1,j] = new_index
        self._clear_hanging_vertex(new_index,[fj,fk])
        #reinit
        self.num_vertices += 1
        self.num_faces += 1
        self._init_adjacency()
        self._init_edges()

