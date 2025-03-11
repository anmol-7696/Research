import torch
import matplotlib.pyplot as plt
import networkx as nx

# Step 1: Define a Simple Graph
# Nodes: 6 (labeled 0-5)
# Edges: Defined in `edge_index`
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 3, 1],  # From
    [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5, 5, 3]   # To
], dtype=torch.long)

# Node features (6 nodes, each with 2 features)
x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]], dtype=torch.float)

# Define graph data
data = Data(x=x, edge_index=edge_index)

# Step 2: Visualize the Graph using matplotlib
def plot_graph(data):
    G = nx.Graph()
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)

    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=1000, font_size=15)
    plt.title("Graph Visualization")
    plt.show()

plot_graph(data)

# Step 3: Define the Graph Neural Network Model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(2, 4)  # First Graph Convolution Layer (Input: 2 features, Output: 4)
        self.conv2 = GCNConv(4, 2)  # Second Graph Convolution Layer (Output: 2 features for classification)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Step 4: Train the GNN
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Dummy labels (for 2-class classification)
y = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
data.y = y  # Assign labels to nodes

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)  # Negative log-likelihood loss
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
for epoch in range(100):
    loss = train()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Step 5: Evaluate Model
model.eval()
pred = model(data).argmax(dim=1)
print("Predicted labels:", pred.tolist())

# Visualizing the final classification
def plot_classification(data, pred):
    G = nx.Graph()
    edge_list = data.edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    color_map = ['red' if label == 0 else 'blue' for label in pred]
    plt.figure(figsize=(6, 6))
    nx.draw(G, with_labels=True, node_color=color_map, edge_color='gray', node_size=1000, font_size=15)
    plt.title("Node Classification Results")
    plt.show()

plot_classification(data, pred)
