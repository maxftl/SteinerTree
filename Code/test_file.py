from TriGraph import *
from visualization import *

F = np.array( [ [0,1,2] ])
V = np.array( [ [0,0],[0,1],[1,0] ] )

graph = TriGraph(V,F)
edge = np.array( [1,1] )
graph.add_edge_to_face(0,edge)
graph.add_edge_to_face(0,np.array([1,-2]))
graph.add_edge_to_face(0,np.array([1,-3]))
#graph.add_edge_to_face(1,np.array([0,1]))
#graph.add_edge_to_face(0,np.array([1,-.5]))
plot_graph(graph)
plt.show()

print(graph.V)
print(graph.F)
print(graph.num_vertices)
print(graph.num_faces)
print(graph.adjacency)