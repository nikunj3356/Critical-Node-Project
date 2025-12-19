# =========================================================
# IMPORTS & SEEDING
# =========================================================
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from sklearn.preprocessing import StandardScaler
from node2vec import Node2Vec
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import random, os, pickle

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
tf.random.set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================================
# GRAPH METRICS (COMMON)
# =========================================================
def largest_connected_component(G):
    if G.number_of_nodes() == 0:
        return G
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()

def effective_graph_resistance(G):
    N = G.number_of_nodes()
    if N <= 1:
        return 0.0
    c = nx.number_connected_components(G)
    if N - c <= 0:
        return np.inf
    L = nx.laplacian_matrix(G).astype(float).todense()
    eig = np.real(np.linalg.eigvals(L))
    nz = eig[eig > 1e-8]
    return np.inf if len(nz) == 0 else (2.0 / (N - c)) * np.sum(1.0 / nz)

def weighted_spectrum(G):
    N = G.number_of_nodes()
    if N <= 1:
        return 0.0
    c = nx.number_connected_components(G)
    if N - c <= 0:
        return np.inf
    L = nx.laplacian_matrix(G).astype(float).todense()
    eig = np.real(np.linalg.eigvals(L))
    nz = eig[eig > 1e-8]
    return np.inf if len(nz) == 0 else np.sum(1.0 / nz)

# =========================================================
# COMMON EVALUATION FUNCTION
# =========================================================
def evaluate_model(G, nodes, scores, name, top_k_percent=20):
    K = max(1, int(len(nodes) * top_k_percent / 100))
    top_nodes = [nodes[i] for i in np.argsort(scores)[::-1][:K]]

    G0 = largest_connected_component(G)
    egr0, ws0 = effective_graph_resistance(G0), weighted_spectrum(G0)

    G1 = G.copy()
    G1.remove_nodes_from(top_nodes)
    G1 = largest_connected_component(G1)
    egr1, ws1 = effective_graph_resistance(G1), weighted_spectrum(G1)

    print(f"\n===== {name} =====")
    print(f"ΔEGR (%) : {((egr1 - egr0) / egr0) * 100:.2f}")
    print(f"ΔWS  (%) : {((ws1 - ws0) / ws0) * 100:.2f}")

# =========================================================
# FEATURE BUILDER (COMMON)
# =========================================================
def build_features(G):
    nodes = sorted(G.nodes())
    deg = np.array([G.degree(n) for n in nodes]).reshape(-1,1)
    clus = np.array([nx.clustering(G,n) for n in nodes]).reshape(-1,1)
    pr = np.array([nx.pagerank(G)[n] for n in nodes]).reshape(-1,1)
    X = StandardScaler().fit_transform(np.hstack([deg, clus, pr]))
    node_map = {n:i for i,n in enumerate(nodes)}
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u,v in G.edges()],
                              dtype=torch.long).t().contiguous()
    return nodes, X, edge_index

# =========================================================
# BASELINE MODELS
# =========================================================
class GCN_Base(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.c1 = GCNConv(d,64)
        self.c2 = GCNConv(64,1)
    def forward(self,x,e):
        return self.c2(F.relu(self.c1(x,e)),e)

class GAT_Base(torch.nn.Module):
    def __init__(self,d):
        super().__init__()
        self.c1 = GATConv(d,32,heads=4)
        self.c2 = GATConv(32*4,1,concat=False)
    def forward(self,x,e):
        return self.c2(F.elu(self.c1(x,e)),e)

class SAGE_Base(torch.nn.Module):
    def __init__(self,d):
        super().__init__()
        self.c1 = SAGEConv(d,64)
        self.c2 = SAGEConv(64,1)
    def forward(self,x,e):
        return self.c2(F.relu(self.c1(x,e)),e)

def run_torch_model(ModelCls, G):
    nodes,X,e = build_features(G)
    data = Data(x=torch.tensor(X,dtype=torch.float), edge_index=e)
    model = ModelCls(X.shape[1])
    model.eval()
    with torch.no_grad():
        scores = model(data.x,data.edge_index).squeeze().numpy()
    return nodes, scores

def run_node2vec(G):
    nodes = sorted(G.nodes())
    n2v = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=1, seed=SEED)
    m = n2v.fit(window=10, min_count=1)
    emb = StandardScaler().fit_transform(np.array([m.wv[str(n)] for n in nodes]))
    scores = np.linalg.norm(emb, axis=1)
    return nodes, scores

# =========================================================
# PROPOSED MODEL: GCN + ILGR
# =========================================================
@tf.keras.utils.register_keras_serializable()
def pairwise_ranking_loss(y_true, y_pred):
    diff = tf.expand_dims(y_true,1) - tf.expand_dims(y_true,0)
    idx = tf.where(diff > 0)
    return tf.cond(tf.shape(idx)[0] > 0,
        lambda: tf.reduce_mean(tf.nn.relu(1.0 - (
            tf.gather(y_pred,idx[:,0]) - tf.gather(y_pred,idx[:,1])
        ))),
        lambda: tf.constant(0.0))

class ILGR:
    def __init__(self,d):
        i = layers.Input(shape=(d,))
        x = layers.Dense(d*4,activation='relu')(i)
        x = layers.Reshape((4,d))(x)
        x = layers.MultiHeadAttention(2,d)(x,x)
        x = layers.Flatten()(x)
        x = layers.Dense(64,activation='relu')(x)
        o = layers.Dense(1,activation='sigmoid')(x)
        self.model = Model(i,o)
        self.model.compile(optimizer='adam', loss=pairwise_ranking_loss)
        self.scaler = StandardScaler()

    def train(self,X,y):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs,y,epochs=50,batch_size=8,verbose=0)

    def predict(self,X):
        return self.model.predict(self.scaler.transform(X)).flatten()

def run_gcn_ilgr(G, critical_nodes):

    embeddings = generate_gcn_embeddings(G)
    nodes = sorted(G.nodes())

    labels = np.array([1 if str(n) in critical_nodes else 0 for n in nodes])

    split = int(0.8 * len(nodes))
    X_train, X_test = embeddings[:split], embeddings[split:]
    y_train, y_test = labels[:split], labels[split:]
    test_nodes = nodes[split:]

    ilgr = ILGR(input_dim=X_train.shape[1])
    ilgr.build_model()
    ilgr.train(X_train, y_train)

    scores = ilgr.predict(X_test)

    return test_nodes, scores

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, output_dim=16):  # Change input_dim to 1
        super(GCN, self).__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim // 2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim // 2)
        self.gcn3 = GCNConv(hidden_dim // 2, output_dim)
        self.bn3 = torch.nn.BatchNorm1d(output_dim)
        self.dropout = torch.nn.Dropout(0.3)
        self.layer_norm = torch.nn.LayerNorm(output_dim)

        self.res_proj = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, x, edge_index):
        x1 = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x1 = self.dropout(x1)
        x2 = F.relu(self.bn2(self.gcn2(x1, edge_index)))
        x2 = self.dropout(x2)
        x3 = self.gcn3(x2, edge_index)  # No log_softmax

        # Project x1 (64 features) to match x3 (16 features)
        # x1_projected = torch.nn.Linear(x1.shape[1], x3.shape[1]).to(x1.device)(x1)

        x1_projected = self.res_proj(x1)
        # Now both have shape (num_nodes, 16), so we can add
        x = x3 + x1_projected
        return x
    
def generate_gcn_embeddings(graph, input_dim=3, hidden_dim=64, output_dim=16, epochs=500, lr=0.0001):
    """Generates node embeddings using GCN with only Degree as input feature."""

    node_map = {node: idx for idx, node in enumerate(graph.nodes())}
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in graph.edges()], dtype=torch.long).t().contiguous()

    num_nodes = len(graph.nodes())
    degree = torch.tensor([graph.degree(node) for node in graph.nodes()], dtype=torch.float).view(-1, 1)
    clustering = torch.tensor([nx.clustering(graph, node) for node in graph.nodes()], dtype=torch.float).view(-1, 1)
    pagerank = torch.tensor([nx.pagerank(graph)[node] for node in graph.nodes()], dtype=torch.float).view(-1, 1)
    scaler = StandardScaler()
    degree = torch.tensor(scaler.fit_transform(degree.numpy()), dtype=torch.float)
    clustering = torch.tensor(scaler.fit_transform(clustering.numpy()), dtype=torch.float)
    pagerank = torch.tensor(scaler.fit_transform(pagerank.numpy()), dtype=torch.float)

    x = torch.cat((degree, clustering, pagerank), dim=1)

    if edge_index.shape[0] != 2:
        raise ValueError(f"Incorrect edge_index shape: {edge_index.shape}, expected (2, num_edges)")

    data = Data(x=x, edge_index=edge_index)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(input_dim, hidden_dim, output_dim).to(device)  # Ensure input_dim=1
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        projection = torch.nn.Linear(out.shape[1], data.x.shape[1]).to(out.device)  # Match 16 → 3
        projected_out = projection(out)

        loss = F.mse_loss(projected_out, data.x)
        #loss = 1 - F.cosine_similarity(projected_out, data.x, dim=1).mean()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        embeddings = model(data.x, data.edge_index).cpu().numpy()

    return embeddings

@tf.keras.utils.register_keras_serializable()
def pairwise_ranking_loss(y_true, y_pred):
    margin = 1.0  # Hyperparameter

    # Pairwise differences: compare every (i, j) pair where y_true[i] > y_true[j]
    pairwise_differences = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
    valid_pairs = tf.where(pairwise_differences > 0)  # Find valid (i, j) pairs

    def compute_loss():
        y_pred_i = tf.gather(y_pred, valid_pairs[:, 0])
        y_pred_j = tf.gather(y_pred, valid_pairs[:, 1])
        return tf.reduce_mean(tf.nn.relu(margin - (y_pred_i - y_pred_j)))

    loss = tf.cond(tf.shape(valid_pairs)[0] > 0, compute_loss, lambda: tf.constant(0.0))
    return loss

class ILGR:
    def __init__(self, input_dim, regression_layers=[256, 128, 64, 1], num_heads=2):
        self.input_dim = input_dim
        self.regression_layers = regression_layers
        self.num_heads = num_heads
        self.model = None
        self.scaler = StandardScaler()

    def build_model(self):
        inputs = layers.Input(shape=(self.input_dim,))

        # Project input into a sequence of tokens for Multi-Head Attention
        x = layers.Dense(self.input_dim * 4, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)  # Expanding to create multiple tokens
        x = layers.Reshape((4, self.input_dim))(x)  # Reshaping into (batch_size, 4, input_dim)
        x = layers.Dropout(0.5)(x) 
        # Multi-Head Self-Attention Layer
        x = layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=self.input_dim)(x, x)
        x = layers.Dropout(0.3)(x)  # Add after MultiHeadAttention
        x = layers.Flatten()(x)  # Flatten back to (batch_size, feature_dim)

        # Fully Connected Layers
        for units in self.regression_layers[:-1]:
            x = layers.Dense(units, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
            x = layers.Dropout(0.5)(x) 
        # output = layers.Dense(self.regression_layers[-1], activation='linear')(x)

        output = layers.Dense(self.regression_layers[-1], activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=output)
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse', metrics=['mae'])
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=pairwise_ranking_loss,  # Focal Loss
            metrics=['accuracy']
        )
        # self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  
        #                 loss=tfa.losses.SigmoidFocalCrossEntropy(alpha=0.75, gamma=2.0),  
        #                 metrics=['accuracy'])
        return self.model

    # def train(self, X, y, epochs=1000, batch_size=8):
    #     X = self.scaler.fit_transform(X)
    #     # self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
    #     class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y)
    #     class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    #     self.model.fit(X, y, epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict)

    def train(self, X, y, epochs=1000, batch_size=8):
        X = self.scaler.fit_transform(X)
        class_weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y)
        class_weights[1] *= 5
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=pairwise_ranking_loss,  # Focal Loss
            metrics=['accuracy']
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size, class_weight=class_weight_dict,callbacks=[early_stopping])

    def predict(self, X):
        X = self.scaler.transform(X)
        return np.clip(self.model.predict(X).flatten(), 0.0, 1.0)

    def save_model(self, filename):
        self.model.save(filename + ".keras")  # Save in Keras format
        with open(filename + "_scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self, filename):
        self.model = tf.keras.models.load_model(filename + ".keras", custom_objects={"pairwise_ranking_loss": pairwise_ranking_loss})  # Match the extension
        with open(filename + "_scaler.pkl", 'rb') as f:
            self.scaler = pickle.load(f)

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    G = nx.read_edgelist("datasetfinal4train.txt", nodetype=str)
    critical = set(open("criticalfinal4train.txt").read().split())

    models = [
        ("GCN Baseline", lambda: run_torch_model(GCN_Base,G)),
        ("GAT Baseline", lambda: run_torch_model(GAT_Base,G)),
        ("GraphSAGE Baseline", lambda: run_torch_model(SAGE_Base,G)),
        ("Node2Vec Baseline", lambda: run_node2vec(G)),
        ("GCN-ILGR (Proposed)", lambda: run_gcn_ilgr(G,critical))
    ]

    for name,fn in models:
        nodes,scores = fn()
        evaluate_model(G,nodes,scores,name)
