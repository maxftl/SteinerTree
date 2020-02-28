import numpy as np
import gurobipy as gp
from gurobipy import GRB
from TriGraph import TriGraph
from SubmodularFunction import SubmodularFunction
import math
from numpy import dot
from numpy.linalg import norm

class Violation:
    def __init__(self, a, b, dir_d, error):
        self.a = a
        self.b = b
        self.dir_d = dir_d
        self.error = error

def calculate_min_flow(graph, objective, supply, num_goods):
    result = {}
    try:
        model = gp.Model("Network optimization")
        model.setParam(GRB.Param.OutputFlag,False)
        flow_indices = gp.tuplelist((i,j) for i in range(graph.num_edges) for j in range(num_goods))
        pos_flow = model.addVars(flow_indices, name = 'pf')
        neg_flow = model.addVars(flow_indices, name = 'nf')
        mu_indices = gp.tuplelist(range(graph.num_edges))
        pos_mu = model.addVars(mu_indices, name = 'p_mu')
        neg_mu = model.addVars(mu_indices, name = 'n_mu')
        subdiff = objective.subdifferential0()
        #Add mu >= f(x)
        for i in range(graph.num_edges):
            (v,w) = graph.edges[i]
            length = graph.edge_lengths[v][w]
            for a in subdiff:
                pos_eq = 0
                neg_eq = 0
                for j in range(num_goods):
                    pos_eq += length * a[j] * pos_flow[i,j]
                    neg_eq += length * a[j] * neg_flow[i,j]
                model.addConstr(pos_mu[i] >= pos_eq)
                model.addConstr(neg_mu[i] >= neg_eq)

        #Add divergence constraints
        div_constraints = [ [] for _ in range(graph.num_vertices)]
        for k in range(graph.num_vertices):
            for j in range(num_goods):
                eq = 0
                for i in graph.vertex_to_edges[k]:
                    edge = graph.edges[i]
                    if k==edge[0]:
                        eq += pos_flow[i,j] - neg_flow[i,j]
                    else:
                        eq += neg_flow[i,j] - pos_flow[i,j]
                div_constraints[k].append( model.addConstr(eq == supply[k][j]) )

        #Define objective
        eq = 0
        for i in range(graph.num_edges):
            eq += pos_mu[i]+neg_mu[i]
        model.setObjective(eq, GRB.MINIMIZE)
        model.optimize()

        result['flow'] = np.array( [ [pos_flow[i,j].x-neg_flow[i,j].x for j in range(num_goods)] for i in range(graph.num_edges) ])
        result['phi'] = np.array( [ [div_constraints[k][j].pi for j in range(num_goods) ] for k in range(graph.num_vertices) ])
        result['value'] = model.ObjVal
        return result
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))
    except AttributeError:
        print('Attribute error (WTF??)')


def calculate_violations(dphi, subgradient_inequalities):
    return [ Violation(a,b,dot(a,dphi),(norm(dot(a,dphi))-b)/norm(a) ) for (a,b) in subgradient_inequalities if norm(a) > 0.1 ]


'''returns indices of faces containing a vertex with non-zero flow'''
def get_used_face_indices(graph, flow):
    result = set()
    for j in range(graph.num_edges):
        f = flow[j,:]
        if np.linalg.norm(f) < 0.1:
            continue
        result |= {i for i in range(graph.num_faces) if len( set(graph.F[i,:]) & set(graph.edges[j]) ) >= 1 }
    return result

