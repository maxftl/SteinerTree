import numpy as np
from TriGraph import TriGraph

class TopoGraph:

    def __init__(self, tgraph, flow, supply, cost_function):
        important_vertices = self._mark_important_vertices(tgraph,flow,supply)
        self._reduced_adjacency(tgraph, flow, supply, important_vertices, cost_function)

    def _mark_important_vertices(self, tgraph, flow, supply):
        is_important = [False for _ in range(tgraph.num_vertices)]
        for i in range(tgraph.num_vertices):
            print(supply[i])
            if np.linalg.norm(supply[i]) >= 0.1:
                is_important[i] = True
                continue
            vertex_flow = [f for f in flow[tgraph.vertex_to_edges[i],:] if np.linalg.norm(f) >= 0.1]
            if len(vertex_flow) >= 3:
                is_important[i] = True
                continue
        return is_important


    def _reduced_adjacency(self, tgraph, flow, supply, is_important, cost_function):
        edges = np.array(tgraph.edges)
        not_deleted = np.ones(len(edges), dtype=np.bool)
        for i in range(len(edges)):
            if not not_deleted[i]:
                continue
            if np.linalg.norm(flow[i]) <= 0.1:
                not_deleted[i] = False
                continue
            for j in range(2):
                while not is_important[edges[i][j]]:
                    adj_edge = [f for f in tgraph.vertex_to_edges[edges[i][j]] if f != i][0]
                    if edges[adj_edge][0] != edges[i][j]:
                        edges[i][j] = edges[adj_edge][0]
                    else:
                        edges[i][j] = edges[adj_edge][1]
                    not_deleted[adj_edge] = False

        edges = edges[not_deleted]
        vi = dict([ (v,j) for (j,v) in enumerate( set(edges.flatten()) ) ])
        iv = dict(enumerate(set(edges.flatten())))
        map_func = np.vectorize(lambda x: vi[x])
        self.edges = map_func(edges)
        self.flow = flow[not_deleted]
        self.supply = np.array([ supply[iv[j]] for j in range(len(iv))])
        self.fixed = np.array([np.linalg.norm(s) >= 0.1 for s in self.supply], dtype = np.bool)
        self.num_vertices = len(vi)
        self.num_edges = len(self.edges)
        self.positions = np.array([ tgraph.V[iv[j]] for j in range(self.num_vertices) ])
        self.costs = np.array([cost_function.data[tuple(f)] for f in np.rint(np.abs(self.flow)).astype(np.int)])
        

