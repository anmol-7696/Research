import matplotlib.pyplot as plt
import networkx as nx

# Create a graph
G = nx.Graph()

# Add nodes
nodes = range(1, 6)
G.add_nodes_from(nodes)

# Add edges (connections)
edges = [(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)]
G.add_edges_from(edges)

# Draw the graph
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G)  # Layout for positioning nodes
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=1000, font_size=14)

# Show plot
plt.title("Graph Neural Network Structure")
plt.show()
