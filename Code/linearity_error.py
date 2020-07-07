from sklearn.linear_model import LinearRegression
from TriGraph import *

def linearity_errors(graph, phi):
    reg = LinearRegression().fit(graph.V,phi)
    u = 0


V = np.array([[1,0],[0,1],[2,3]])
F = np.array([[0,1,2]])
graph = TriGraph(V,F)
phi = [1,2,3]
linearity_errors(graph,phi)