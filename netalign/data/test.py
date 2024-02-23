import networkx as nx
from itertools import chain

# Create a sample graph
G = nx.Graph()
G.add_node(1, color='red', size=10)
G.add_node(2, color='blue', size=20)
G.add_node(3, color='green', size=15)

G.add_edge(1, 2, weight=5, label='A')
G.add_edge(2, 3, weight=8, label='B')
G.add_edge(1, 3, weight=3, label='C')

# Get node/edge attribute names
edge_attributes = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
node_attributes = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))

print("Node Attributes:", node_attributes)
print("Edge Attributes:", edge_attributes)
