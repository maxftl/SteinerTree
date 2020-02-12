from submodular_network_flow import *
import numpy as np
from SubmodularFunction import *
from TriGraph import *
from math import sqrt
from visualization import *

#Standard example with 3 vertices
q1 = 0
q2 = 0
V = np.array([ [0,-1],[0,0],[0,1], [2,0] ])
F = np.array([ [0,1,3],[1,2,3] ], dtype = np.int)
graph = TriGraph(V,F)
objective_data = {
    (0,0,0): 0,
    (0,1,0): 1,
    (1,0,0): 1,
    (0,0,1): 1,
    (1,1,0): 1+q1,
    (1,0,1): 1+q1,
    (0,1,1): 1+q1,
    (1,1,1): 1+q1+q2
}
objective = SubmodularFunction(objective_data, 3)
supply = np.array( [ [1,0,0],[0,1,0], [0,0,1], [-1,-1,-1] ] )
result = calculate_min_flow(graph, objective, supply, 3)
DPhi = graph.element_wise_differential(result['phi'].T)
subgradient_inequalities = objective.subdifferential0_inequalities()
violations = [calculate_violations(d, subgradient_inequalities) for d in DPhi]
max_violations = [ sorted(v,key=lambda x: -x.error)[0] for v in violations ]
print([v.error for v in max_violations])

plot_graph(graph)
plot_flow(graph, result['flow'])
plot_violations(graph, [v.dir_d for v in max_violations])
plt.show()