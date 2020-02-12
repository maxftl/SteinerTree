from submodular_network_flow import *
import numpy as np
from SubmodularFunction import *
from TriGraph import *
from math import sqrt
from visualization import *

#Standard example with 3 vertices
l = 3
h = sqrt(l**2-1)

V = np.array([ [0,-1],[0,1],[h,0] ])
F = np.array([ [0,1,2] ], dtype = np.int)
graph = TriGraph(V,F)
objective_data = {
    (0,0): 0,
    (0,1): 1,
    (1,0): 1,
    (1,1): 2
}
objective = SubmodularFunction(objective_data, 2)
supply = np.array( [ [1,0],[0,1], [-1,-1] ] )
result = calculate_min_flow(graph, objective, supply, 2)
DPhi = graph.element_wise_differential(result['phi'].T)
subgradient_inequalities = objective.subdifferential0_inequalities()
violations = [calculate_violations(d, subgradient_inequalities) for d in DPhi]
max_violations = [ sorted(v,key=lambda x: -x.error)[0] for v in violations ]
print([v.error for v in max_violations])

plot_graph(graph)
plot_flow(graph, result['flow'])
plot_violations(graph, [v.dir_d for v in max_violations])
plt.show()