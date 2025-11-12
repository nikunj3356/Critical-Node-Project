import networkx as nx

# Load the edge list from the file
file_path = "datasetfinal41.txt"

# Read edges from the file
edges = []
with open(file_path, "r") as file:
    for line in file:
        u, v = map(int, line.strip().split())  # Convert to integers
        edges.append((u, v))

# Create a NetworkX graph
G = nx.Graph()
G.add_edges_from(edges)

# Compute centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

# Combine centrality scores
centrality_scores = {}
for node in G.nodes():
    centrality_scores[node] = (
        degree_centrality[node] +
        betweenness_centrality[node] +
        closeness_centrality[node] +
        eigenvector_centrality[node]
    )

# Sort nodes by importance
sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)

# Extract top 5% critical nodes
top_5_percent = int(0.05 * len(G.nodes()))
critical_nodes = sorted_nodes[:top_5_percent]

# Print the critical nodes
print("Top 5% Critical Nodes:")
for node, score in critical_nodes:
    print(f"{node}")