#!/usr/bin/env python3
"""
Simple Temporal Graph Neural Network implementation for fraud detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import time
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class SimpleTGN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super(SimpleTGN, self).__init__()
        
        # Graph convolution layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Temporal components
        self.temporal_embedding = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x, edge_index, batch=None):
        # Graph convolution
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        
        # Global pooling for graph-level representation
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        graph_embedding = global_mean_pool(h2, batch)
        
        # For simplicity, we'll skip the temporal LSTM in this basic version
        # In a full TGN, you'd process temporal sequences here
        
        # Classification
        output = self.classifier(graph_embedding)
        
        return output

class TGNTrainer:
    def __init__(self, input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SimpleTGN(input_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = nn.BCELoss()
        
        print(f"Using device: {device}")
    
    def prepare_graph_data(self, X, y, edges, edge_features, transaction_ids, node_mapping):
        """Convert tabular data to graph format"""
        graph_data_list = []
        
        # Create mapping from transaction ID to features
        id_to_features = {}
        id_to_label = {}
        
        for i, tid in enumerate(transaction_ids):
            id_to_features[tid] = X.iloc[i].values
            id_to_label[tid] = y.iloc[i] if hasattr(y, 'iloc') else y[i]
        
        # Group transactions by connected components (subgraphs)
        # For simplicity, we'll create small subgraphs based on time windows
        time_windows = self.create_time_windows(transaction_ids, X, window_size=100)
        
        for window_transactions in time_windows:
            if len(window_transactions) < 2:  # Need at least 2 nodes
                continue
            
            # Create local node mapping for this subgraph
            local_node_map = {tid: i for i, tid in enumerate(window_transactions)}
            
            # Node features
            node_features = []
            node_labels = []
            
            for tid in window_transactions:
                if tid in id_to_features:
                    node_features.append(id_to_features[tid])
                    node_labels.append(id_to_label[tid])
            
            if len(node_features) < 2:
                continue
            
            node_features = np.array(node_features)
            node_labels = np.array(node_labels)
            
            # Create edges for this subgraph (connect transactions in temporal order)
            subgraph_edges = []
            for i in range(len(window_transactions) - 1):
                subgraph_edges.append([i, i + 1])  # Connect consecutive transactions
            
            if len(subgraph_edges) == 0:
                continue
            
            # Create PyTorch Geometric Data object
            data = Data(
                x=torch.FloatTensor(node_features),
                edge_index=torch.LongTensor(subgraph_edges).t().contiguous(),
                y=torch.FloatTensor(node_labels)
            )
            
            graph_data_list.append(data)
        
        return graph_data_list
    
    def create_time_windows(self, transaction_ids, X, window_size=100):
        """Create time windows for subgraph construction"""
        # For simplicity, create windows based on transaction order
        windows = []
        transaction_list = list(transaction_ids)
        
        for i in range(0, len(transaction_list), window_size):
            window = transaction_list[i:i + window_size]
            windows.append(window)
        
        return windows
    
    def train_epoch(self, train_data_list):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        # Shuffle training data
        np.random.shuffle(train_data_list)
        
        for data in train_data_list:
            data = data.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            out = self.model(data.x, data.edge_index)
            out = out.squeeze()
            
            # Calculate loss (handle different tensor shapes)
            if len(out.shape) == 0:
                out = out.unsqueeze(0)
            if len(data.y.shape) == 0:
                target = data.y.unsqueeze(0)
            else:
                target = data.y
            
            # Take mean if multiple nodes per graph
            if len(out) != len(target):
                if len(out) > len(target):
                    out = out[:len(target)]
                else:
                    target = target[:len(out)]
            
            loss = self.criterion(out, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_data_list)
    
    def evaluate(self, val_data_list, batch_size=32):
        """Evaluate model and measure inference time"""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        inference_times = []
        
        with torch.no_grad():
            for i in range(0, len(val_data_list), batch_size):
                batch_data = val_data_list[i:i + batch_size]
                
                # Measure inference time
                start_time = time.time()
                
                for data in batch_data:
                    data = data.to(self.device)
                    
                    # Forward pass
                    out = self.model(data.x, data.edge_index)
                    out = out.squeeze()
                    
                    if len(out.shape) == 0:
                        out = out.unsqueeze(0)
                    if len(data.y.shape) == 0:
                        target = data.y.unsqueeze(0)
                    else:
                        target = data.y
                    
                    # Take mean prediction if multiple nodes
                    if len(out) > 1:
                        prediction = out.mean().cpu().numpy()
                        label = target.float().mean().cpu().numpy()
                    else:
                        prediction = out.cpu().numpy().item()
                        label = target.cpu().numpy().item()
                    
                    all_predictions.append(prediction)
                    all_labels.append(label)
                
                batch_time = time.time() - start_time
                inference_times.append(batch_time)
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Handle edge cases
        if len(np.unique(all_labels)) < 2:
            print("Warning: Only one class in validation set")
            auc_roc = 0.5
            auc_pr = all_labels.mean()
        else:
            auc_roc = roc_auc_score(all_labels, all_predictions)
            precision, recall, _ = precision_recall_curve(all_labels, all_predictions)
            auc_pr = auc(recall, precision)
        
        # Average inference time per transaction
        avg_inference_time = np.mean(inference_times) / batch_size * 1000  # ms per transaction
        
        return {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'inference_time_ms': avg_inference_time,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self, train_data_list, val_data_list, epochs=20):
        """Full training loop"""
        print(f"Training TGN with {len(train_data_list)} training graphs...")
        
        start_time = time.time()
        best_auc = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_data_list)
            
            # Validate every 5 epochs
            if epoch % 5 == 0:
                val_results = self.evaluate(val_data_list)
                current_auc = val_results['auc_roc']
                
                print(f"Epoch {epoch}: Loss={train_loss:.4f}, Val AUC={current_auc:.4f}")
                
                if current_auc > best_auc:
                    best_auc = current_auc
                    torch.save(self.model.state_dict(), 'results/best_tgn_model.pt')
        
        training_time = time.time() - start_time
        
        # Load best model and final evaluation
        self.model.load_state_dict(torch.load('results/best_tgn_model.pt'))
        final_results = self.evaluate(val_data_list)
        final_results['training_time'] = training_time
        
        print(f"\nTGN Training Complete!")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Final AUC-ROC: {final_results['auc_roc']:.4f}")
        print(f"Final AUC-PR: {final_results['auc_pr']:.4f}")
        print(f"Inference time: {final_results['inference_time_ms']:.2f} ms per transaction")
        
        return final_results

def run_tgn_experiment():
    """Run the complete TGN experiment"""
    print("=== Running TGN Experiment ===")
    
    # Load processed data
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    edges = data['edges']
    edge_features = data['edge_features']
    train_ids = data['train_ids']
    val_ids = data['val_ids']
    node_mapping = data['node_mapping']
    
    # Initialize TGN trainer
    input_dim = X_train.shape[1]
    trainer = TGNTrainer(input_dim)
    
    # Prepare graph data
    print("Preparing graph data...")
    train_graphs = trainer.prepare_graph_data(X_train, y_train, edges, edge_features, train_ids, node_mapping)
    val_graphs = trainer.prepare_graph_data(X_val, y_val, edges, edge_features, val_ids, node_mapping)
    
    print(f"Created {len(train_graphs)} training graphs and {len(val_graphs)} validation graphs")
    
    if len(train_graphs) == 0 or len(val_graphs) == 0:
        print("Error: No graph data created. Check data preprocessing.")
        return None
    
    # Train TGN
    results = trainer.train(train_graphs, val_graphs, epochs=20)
    
    # Save results
    with open('results/tgn_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    results = run_tgn_experiment()
