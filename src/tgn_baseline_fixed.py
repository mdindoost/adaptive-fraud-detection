#!/usr/bin/env python3
"""
Fixed Simple TGN implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import time
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

class SimpleTGN(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1):  # Smaller for speed
        super(SimpleTGN, self).__init__()
        
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index, batch=None):
        h1 = F.relu(self.conv1(x, edge_index))
        h2 = F.relu(self.conv2(h1, edge_index))
        
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        graph_embedding = global_mean_pool(h2, batch)
        output = self.classifier(graph_embedding)
        
        return output

class SimpleTGNTrainer:
    def __init__(self, input_dim, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = SimpleTGN(input_dim).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()
    
    def create_simple_graphs(self, X, y, graph_size=50):
        """Create simple graph data for validation"""
        graphs = []
        
        for i in range(0, len(X), graph_size):
            end_idx = min(i + graph_size, len(X))
            if end_idx - i < 10:  # Skip small graphs
                continue
                
            # Node features
            node_features = X.iloc[i:end_idx].values
            node_labels = y.iloc[i:end_idx].values
            
            # Simple edges (chain graph)
            edges = []
            for j in range(len(node_features) - 1):
                edges.append([j, j + 1])
            
            if len(edges) == 0:
                continue
            
            # Create graph
            data = Data(
                x=torch.FloatTensor(node_features),
                edge_index=torch.LongTensor(edges).t().contiguous(),
                y=torch.FloatTensor([node_labels.mean()])  # Graph-level label
            )
            
            graphs.append(data)
        
        return graphs
    
    def train_simple(self, X_train, y_train, X_val, y_val, epochs=10):
        """Simplified training for quick validation"""
        print("Creating simple graph structures...")
        
        train_graphs = self.create_simple_graphs(X_train, y_train)
        val_graphs = self.create_simple_graphs(X_val, y_val)
        
        print(f"Created {len(train_graphs)} train graphs, {len(val_graphs)} val graphs")
        
        if len(train_graphs) == 0 or len(val_graphs) == 0:
            return {
                'auc_roc': 0.5,
                'auc_pr': y_val.mean(),
                'training_time': 0,
                'inference_time_ms': 100,
                'error': 'No graphs created'
            }
        
        start_time = time.time()
        
        # Training
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for data in train_graphs:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                
                out = self.model(data.x, data.edge_index)
                loss = self.criterion(out.squeeze(), data.y.unsqueeze(0) if data.y.dim() == 0 else data.y)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / len(train_graphs)
                print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluation
        self.model.eval()
        predictions = []
        labels = []
        inference_times = []
        
        with torch.no_grad():
            for data in val_graphs:
                data = data.to(self.device)
                
                start_inf = time.time()
                out = self.model(data.x, data.edge_index)
                inf_time = time.time() - start_inf
                
                pred = out.squeeze().cpu().numpy()
                if np.isscalar(pred):
                    predictions.append(pred)
                else:
                    predictions.append(pred.item() if pred.size == 1 else pred.mean())
                
                label = data.y.cpu().numpy()
                if np.isscalar(label):
                    labels.append(label)
                else:
                    labels.append(label.item() if label.size == 1 else label.mean())
                
                inference_times.append(inf_time)
        
        # Calculate metrics
        if len(set(labels)) > 1:
            auc_roc = roc_auc_score(labels, predictions)
            precision, recall, _ = precision_recall_curve(labels, predictions)
            auc_pr = auc(recall, precision)
        else:
            auc_roc = 0.5
            auc_pr = np.mean(labels)
        
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        
        results = {
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'training_time': training_time,
            'inference_time_ms': avg_inference_time,
            'predictions': predictions,
            'labels': labels
        }
        
        print(f"\n🕸️ Simple TGN Results:")
        print(f"   AUC-ROC: {auc_roc:.4f}")
        print(f"   AUC-PR: {auc_pr:.4f}")
        print(f"   Training time: {training_time:.2f} seconds")
        print(f"   Inference time: {avg_inference_time:.2f} ms per transaction")
        
        return results

def run_simple_tgn():
    """Run simplified TGN experiment"""
    print("=== Running Simple TGN Experiment ===")
    
    # Load data
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Train simple TGN
    trainer = SimpleTGNTrainer(X_train.shape[1])
    results = trainer.train_simple(X_train, y_train, X_val, y_val)
    
    # Save results
    with open('results/simple_tgn_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    return results

if __name__ == "__main__":
    results = run_simple_tgn()
