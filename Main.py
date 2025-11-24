import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model
import pickle
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy import stats
import os
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seeds for reproducibility
SEED = 42
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tf.random.set_seed(seed)

set_seed(SEED)

# def find_best_threshold(y_true, y_scores):
#     precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
#     # Find threshold where F1-score is maximized
#     f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
#     best_threshold = thresholds[np.argmax(f1_scores)]
    
#     print(f"Best Threshold Found: {best_threshold:.4f}")
#     return best_threshold
# @tf.keras.utils.register_keras_serializable()
# def pairwise_ranking_loss(y_true, y_pred):
#     margin = 1.0  # Hyperparameter

#     # Pairwise differences: we compare every (i, j) pair where y_true[i] > y_true[j]
#     pairwise_differences = tf.expand_dims(y_true, 1) - tf.expand_dims(y_true, 0)
#     valid_pairs = tf.where(pairwise_differences > 0)  # Find valid (i, j) pairs

#     # Extract predictions for the valid pairs
#     y_pred_i = tf.gather(y_pred, valid_pairs[:, 0])
#     y_pred_j = tf.gather(y_pred, valid_pairs[:, 1])

#     # Compute ranking loss only for valid pairs
#     loss = tf.reduce_mean(tf.nn.relu(margin - (y_pred_i - y_pred_j)))
#     return loss

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
def find_best_threshold(y_true, y_scores):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

    # Compute F1-Scores for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    # Get best threshold but enforce a minimum value (to prevent 0.0)
    best_threshold = max(thresholds[np.argmax(f1_scores)], 0.02)  # Ensures threshold isn't 0
    print(f"Best Threshold Found: {best_threshold:.4f}")
    return best_threshold
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

    def forward(self, x, edge_index):
        x1 = F.relu(self.bn1(self.gcn1(x, edge_index)))
        x1 = self.dropout(x1)
        x2 = F.relu(self.bn2(self.gcn2(x1, edge_index)))
        x2 = self.dropout(x2)
        x3 = self.gcn3(x2, edge_index)  # No log_softmax

        # Project x1 (64 features) to match x3 (16 features)
        x1_projected = torch.nn.Linear(x1.shape[1], x3.shape[1]).to(x1.device)(x1)

        # Now both have shape (num_nodes, 16), so we can add
        x = x3 + x1_projected
        return x

# Generate node embeddings using GCN
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

# ILGR Model with Self-Attention
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

# Load graph and critical nodes
def load_graph(file_path):
    return nx.read_edgelist(file_path, nodetype=str)

def load_critical_nodes(file_path):
    with open(file_path, 'r') as f:
        return set(f.read().split())

# Prepare training data
def prepare_training_data(graphs, critical_nodes_list):
    all_embeddings, all_labels = [], []
    for graph, critical_nodes in zip(graphs, critical_nodes_list):
        embeddings = generate_gcn_embeddings(graph)
        labels = np.array([1 if str(node) in critical_nodes else 0 for node in graph.nodes()])
        all_embeddings.append(embeddings)
        all_labels.append(labels)
    return np.vstack(all_embeddings), np.concatenate(all_labels)

# Main function
def main():
    graph_path = "datasetfinal1train.txt"
    critical_path = "criticalfinal1train.txt"

    graph = load_graph(graph_path)
    critical_nodes = load_critical_nodes(critical_path)

    # 80/20 node split
    nodes = list(graph.nodes())
    np.random.shuffle(nodes)
    split_idx = int(len(nodes) * 0.8)
    train_nodes = set(nodes[:split_idx])
    test_nodes = set(nodes[split_idx:])

    # Create embeddings once
    embeddings = generate_gcn_embeddings(graph)

    # Prepare labels
    labels = np.array([1 if str(node) in critical_nodes else 0 for node in graph.nodes()])
    node_index = {node: i for i, node in enumerate(graph.nodes())}

    train_idx = [node_index[n] for n in train_nodes]
    test_idx  = [node_index[n] for n in test_nodes]

    train_embeddings = embeddings[train_idx]
    test_embeddings  = embeddings[test_idx]
    train_labels = labels[train_idx]
    test_labels  = labels[test_idx]

    model_filename = "ilgr_trained"

    # Train / Load model
    if os.path.exists(model_filename + ".keras"):
        print("Loading existing model...")
        ilgr = ILGR(input_dim=train_embeddings.shape[1])
        ilgr.load_model(model_filename)
    else:
        print("Training new ILGR model...")
        ilgr = ILGR(input_dim=train_embeddings.shape[1])
        ilgr.build_model()
        ilgr.train(train_embeddings, train_labels, epochs=1000)
        ilgr.save_model(model_filename)

    predicted_scores = ilgr.predict(test_embeddings)

    threshold = np.percentile(predicted_scores, 95)
    predicted_binary = (predicted_scores >= threshold).astype(int)

    precision = precision_score(test_labels, predicted_binary)
    recall = recall_score(test_labels, predicted_binary)
    f1 = f1_score(test_labels, predicted_binary)
    accuracy = accuracy_score(test_labels, predicted_binary)

    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    # Print unique predicted classes and their counts
    # y_pred = np.array(predicted_labels)

    # Print unique predicted classes and their counts
    # unique_classes, counts = np.unique(y_pred, return_counts=True)
    # print("Predicted Classes:", unique_classes)
    # print("Counts:", counts)
    # print(f"Critical Nodes Found: {sum(test_labels)}")
    # print("First 20 Labels:", test_labels[:20])
if __name__ == "__main__":
    main()