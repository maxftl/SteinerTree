from submodular_network_flow import *
import numpy as np
from SubmodularFunction import *
from TriGraph import *
from math import sqrt
from visualization import *
from example_library import *

#Standard example with 3 vertices

max_error = 0.1
max_iterations = 100

example = ex_shortest_path_2d()

graph = example['graph']
objective = example['objective']
supply = example['supply']
dim = example['dim']

subgradient_inequalities = objective.subdifferential0_inequalities()

for _ in range(max_iterations):
    result = calculate_min_flow(graph, objective, supply, dim)
    DPhi = graph.element_wise_differential(result['phi'].T)
    violations = [calculate_violations(d, subgradient_inequalities) for d in DPhi]
    max_violations = [ sorted(v,key=lambda x: -x.error)[0] for v in violations ]
    print([v.error for v in max_violations])
    _, max_error_index = max( (v.error,i) for i,v in enumerate(max_violations) )
    plot_graph(graph)
    plot_flow(graph, result['flow'])
    plot_violations(graph, [v.dir_d for v in max_violations])
    plt.show()
    if max([v.error for v in max_violations]) <= max_error:
        print("Error < max_error, done")
        break
    graph.add_edge_to_face(max_error_index,max_violations[max_error_index].dir_d)
    print(graph.F)
    print(graph.V)
    supply = np.vstack( (supply,np.zeros(dim)) )
    