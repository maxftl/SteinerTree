import numpy as np


def armijo_rule(f_func, grad_val, X, alpha = 1e-4, initial_step_size = 1):
    initial_f = f_func(X)
    grad_norm = np.sum(grad_val**2)
    while f_func(X - initial_step_size*grad_val) > initial_f - initial_step_size*alpha*grad_norm:
        initial_step_size *= 0.5
    return initial_step_size


def optimize_fixed_flow(topo_graph, eps = .01, precision = 1e-5):
    mu = eps
    mu2 = mu**2
    relaxed_norm = lambda z: np.sqrt(np.linalg.norm(z,axis=1)**2 + mu2) - mu
    grad_relaxed_norm = lambda z: np.power(np.linalg.norm(z,axis=1)**2 + mu2,-.5).reshape((z.shape[0],1)) * z
    n_iterations = 1000
    #Build function matrices
    D = np.zeros((topo_graph.num_edges,topo_graph.num_vertices))
    C = np.zeros((topo_graph.num_vertices,topo_graph.num_edges))
    for (j,e) in enumerate(topo_graph.edges):
        D[j,e[0]] = 1
        D[j,e[1]] = -1
        if not topo_graph.fixed[e[0]]:
            C[e[0],j] = topo_graph.costs[j]
        if not topo_graph.fixed[e[1]]:
            C[e[1],j] = -topo_graph.costs[j]
    #Build function
    X0 = np.copy(topo_graph.positions)
    f = lambda X: np.dot(topo_graph.costs,relaxed_norm(np.dot(D,X)))
    grad_f = lambda X: np.dot(C,grad_relaxed_norm(np.dot(D,X)))
    for iteration in range(n_iterations):
        grad_val = grad_f(X0)
        step_size = armijo_rule(f,grad_val,X0)
        X0 -= step_size*grad_val
        if np.linalg.norm(grad_val) < precision:
            print('Optimized vertex positions in ' + str(iteration+1) + ' iterations.')
            break
    topo_graph.positions = X0


'''
num_edges = 3
num_vertices = 4
edges = np.array([ [0,1],[1,2],[1,3] ])
eps = .1
costs = np.array([1,1,1])
positions = np.array([ [0,-1],[0,0],[0,1],[5,0] ], dtype = np.float)
fixed = np.array([True,False,True,True],dtype = np.bool)

mu = eps
mu2 = mu**2
L = np.sqrt(2)/mu
relaxed_norm = lambda z: np.sqrt(np.linalg.norm(z,axis=1)**2 + mu2) - mu
grad_relaxed_norm = lambda z: np.power(np.linalg.norm(z,axis=1)**2 + mu2,-.5).reshape((z.shape[0],1)) * z
n_iterations = 1000
#Build function matrices
D = np.zeros((num_edges,num_vertices))
C = np.zeros((num_vertices,num_edges))
for (j,e) in enumerate(edges):
    D[j,e[0]] = 1
    D[j,e[1]] = -1
    if not fixed[e[0]]:
        C[e[0],j] = costs[j]
    if not fixed[e[1]]:
        C[e[1],j] = -costs[j]
#Build function
X0 = np.copy(positions)
f = lambda X: np.dot(costs,relaxed_norm(np.dot(D,X)))
grad_f = lambda X: np.dot(C,grad_relaxed_norm(np.dot(D,X)))
for iteration in range(n_iterations):
    grad_val = grad_f(X0)
    #print(np.linalg.norm(grad_val))
    #print(iteration)
    #print(X0[1,0]-np.tan(np.pi/6))
    step_size = armijo_rule(f,grad_val,X0)
    X0 -= step_size*grad_val

'''