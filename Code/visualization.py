import numpy as np
import matplotlib.pyplot as plt
from TriGraph import *
from numpy.linalg import norm

def plot_graph(graph):
    plt.triplot(graph.V[:,0],graph.V[:,1],graph.F, color = 'gray')

def plot_flow(graph, flow):
    colorwheel = [(1,0,0),(0,.8,0),(0,0,1)]
    eps = 1e-1
    shift_fac = 1e-2
    for i in range(graph.num_edges):
        edge = graph.edges[i]
        normal = graph.V[edge[1]][::-1] - graph.V[edge[0]][::-1]
        normal = normal/np.linalg.norm(normal)
        normal[1] *= -1
        shift = shift_fac*normal
        for j in range(flow.shape[1]):
            if abs(flow[i][j]) < eps:
                continue
            bottom = flow[i][j] > 0
            top = not(bottom)
            start_node = edge[top]
            end_node = edge[bottom]
            x = graph.V[start_node][0]
            y = graph.V[start_node][1]
            dx = graph.V[end_node][0]-x
            dy = graph.V[end_node][1]-y
            plt.arrow(x+j*shift[0],y+j*shift[1],dx,dy, color=colorwheel[j], head_width = .03, alpha = .8)

def plot_violations(graph, violation_directions):
    for (f,d) in zip(graph.F,violation_directions):
        center = (graph.V[f[0],:]+graph.V[f[1],:]+graph.V[f[2],:])/3
        dn = 0.1 * d/norm(d)
        foot = center - 0.5*dn
        plt.arrow(foot[0],foot[1],dn[0],dn[1], head_width = .03)

def mark_violating_face(graph, face_id):
    vertices = graph.V[graph.F[face_id],:]
    plt.fill(vertices[:,0], vertices[:,1], color = 'yellow', alpha = 0.3)

def show_max_error(max_error):
    plt.suptitle('Relative max error: ' + str(max_error))
