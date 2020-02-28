from submodular_network_flow import *
import numpy as np
from SubmodularFunction import *
from TriGraph import *
from math import sqrt
from visualization import *
from example_library import *
from TopoGraph import *
from optimize_fixed_flow import optimize_fixed_flow
from triangulation import *

#Standard example with 3 vertices

max_error = 0.005
max_iterations = 100

example = ex_steiner_tree(4)

graph = example['graph']
objective = example['objective']
supply = example['supply']
dim = example['dim']

weight_by_area = False

subgradient_inequalities = objective.subdifferential0_inequalities()

last_obj_val = 0
for c in range(max_iterations):
    result = calculate_min_flow(graph, objective, supply, dim)
    used_faces = get_used_face_indices(graph, result['flow'])
    #print(result['phi'])
    DPhi = graph.element_wise_differential(result['phi'].T)
    violations = [calculate_violations(d, subgradient_inequalities) for d in DPhi]
    max_violations = [ sorted(v,key=lambda x: -x.error)[0] for v in violations ]
    #print([v.error for v in max_violations])
    for j in range(graph.num_faces):
        if j not in used_faces:
            max_violations[j].error = 0
        elif weight_by_area:
            max_violations[j].error *= np.abs(np.linalg.det(graph.V[graph.F[j,(1,2)],:]-graph.V[graph.F[j,(0,0)],:] ))
    _, max_error_index = max( (v.error,i) for i,v in enumerate(max_violations) )
    plot_graph(graph)
    plot_flow(graph, result['flow'])
    #plot_violations(graph, [v.dir_d for v in max_violations])
    mark_violating_face(graph, max_error_index)
    show_max_error(max_violations[max_error_index].error)
    plt.show()
    if result['value'] <= 0.999*last_obj_val:
        print('Optimize vertex positions')
        topo = TopoGraph(graph,result['flow'],supply,objective)
        optimize_fixed_flow(topo, eps = 0.001)
        graph = triangulate(topo)
        supply = topo.supply
        last_obj_val = 0
        continue
    last_obj_val = result['value']
    if max([v.error for v in max_violations]) <= max_error:
        print("Error < max_error, done")
        break
    graph.add_edge_to_face(max_error_index,max_violations[max_error_index].dir_d)
    #print(graph.F)
    #print(graph.V)
    supply = np.vstack( (supply,np.zeros(dim)) )
    