#!/usr/bin/env python3
"""
Data preprocessing for IEEE-CIS fraud detection dataset
Creates temporal splits and graph structures for comparison
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import os
from datetime import datetime

class FraudDataProcessor:
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_raw_data(self):
        """Load the IEEE-CIS dataset"""
        print("Loading IEEE-CIS dataset...")
        
        # Load transaction data
        train_transaction = pd.read_csv(f"{self.data_path}/train_transaction.csv")
        train_identity = pd.read_csv(f"{self.data_path}/train_identity.csv")
        
        # Merge transaction and identity data
        train_data = train_transaction.merge(train_identity, on='TransactionID', how='left')
        
        print(f"Dataset loaded: {train_data.shape[0]} transactions, {train_data.shape[1]} features")
        print(f"Fraud rate: {train_data['isFraud'].mean():.4f}")
        
        return train_data
    
    def create_temporal_features(self, df):
        """Create temporal features from TransactionDT"""
        df = df.copy()
        
        # TransactionDT is seconds from some reference point
        df['hour'] = (df['TransactionDT'] / 3600) % 24
        df['day'] = (df['TransactionDT'] / (3600 * 24)) % 7
        df['month'] = ((df['TransactionDT'] / (3600 * 24 * 30)) % 12).astype(int)
        
        return df
    
    def preprocess_features(self, df):
        """Clean and preprocess features"""
        df = df.copy()
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Select important features for quick validation
        numerical_features = [
            'TransactionAmt', 'TransactionDT', 'hour', 'day', 'month',
            'card1', 'card2', 'card3', 'card5',
            'addr1', 'addr2', 'C1', 'C2', 'C4', 'C6', 'C8', 'C10', 'C12', 'C14',
            'D1', 'D2', 'D3', 'D4', 'D10', 'D11', 'D15'
        ]
        
        categorical_features = [
            'ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain',
            'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9'
        ]
        
        # Keep only available features
        numerical_features = [f for f in numerical_features if f in df.columns]
        categorical_features = [f for f in categorical_features if f in df.columns]
        
        print(f"Using {len(numerical_features)} numerical and {len(categorical_features)} categorical features")
        
        # Fill missing values
        for col in numerical_features:
            df[col] = df[col].fillna(df[col].median())
            
        for col in categorical_features:
            df[col] = df[col].fillna('Unknown')
            
        # Encode categorical variables
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        
        # Select final feature set
        feature_columns = numerical_features + categorical_features
        X = df[feature_columns].copy()
        y = df['isFraud'].copy()
        
        # Scale numerical features
        num_indices = [i for i, col in enumerate(feature_columns) if col in numerical_features]
        X.iloc[:, num_indices] = self.scaler.fit_transform(X.iloc[:, num_indices])
        
        return X, y, feature_columns, df['TransactionID'], df['TransactionDT']
    
    def create_temporal_split(self, df):
        """Create temporal train/validation split (no data leakage)"""
        # Sort by transaction time
        df_sorted = df.sort_values('TransactionDT').reset_index(drop=True)
        
        # Use first 80% for training, last 20% for validation (temporal split)
        split_point = int(0.8 * len(df_sorted))
        
        train_data = df_sorted.iloc[:split_point]
        val_data = df_sorted.iloc[split_point:]
        
        print(f"Temporal split: {len(train_data)} train, {len(val_data)} validation")
        print(f"Train fraud rate: {train_data['isFraud'].mean():.4f}")
        print(f"Val fraud rate: {val_data['isFraud'].mean():.4f}")
        
        return train_data, val_data
    
    def create_simple_graph_data(self, df):
        """Create simple graph structure for TGN validation"""
        # For quick validation, create edges between transactions with same card
        edges = []
        edge_timestamps = []
        edge_features = []
        
        # Group by card1 (main card identifier)
        card_groups = df.groupby('card1')
        
        node_id_map = {}
        current_node_id = 0
        
        for card, group in card_groups:
            if len(group) < 2:  # Need at least 2 transactions to create edges
                continue
                
            group_sorted = group.sort_values('TransactionDT')
            transactions = group_sorted['TransactionID'].values
            timestamps = group_sorted['TransactionDT'].values
            amounts = group_sorted['TransactionAmt'].values
            
            # Create edges between consecutive transactions for same card
            for i in range(len(transactions) - 1):
                # Map transaction IDs to node IDs
                if transactions[i] not in node_id_map:
                    node_id_map[transactions[i]] = current_node_id
                    current_node_id += 1
                if transactions[i+1] not in node_id_map:
                    node_id_map[transactions[i+1]] = current_node_id
                    current_node_id += 1
                
                # Add edge
                edges.append([node_id_map[transactions[i]], node_id_map[transactions[i+1]]])
                edge_timestamps.append(timestamps[i+1])
                edge_features.append([amounts[i+1], timestamps[i+1] - timestamps[i]])
        
        edges = np.array(edges)
        edge_timestamps = np.array(edge_timestamps)
        edge_features = np.array(edge_features)
        
        print(f"Created graph: {current_node_id} nodes, {len(edges)} edges")
        
        return edges, edge_timestamps, edge_features, node_id_map
    
    def process_and_save(self):
        """Complete preprocessing pipeline"""
        print("=== Starting data preprocessing ===")
        
        # Load raw data
        raw_data = self.load_raw_data()
        
        # Sample for quick validation (use 50k transactions)
        sample_data = raw_data.sample(n=min(50000, len(raw_data)), random_state=42)
        
        # Create temporal split
        train_data, val_data = self.create_temporal_split(sample_data)
        
        # Preprocess features
        print("\nProcessing training data...")
        X_train, y_train, feature_cols, train_ids, train_times = self.preprocess_features(train_data, is_training=True)
        
        print("Processing validation data...")
        X_val, y_val, _, val_ids, val_times = self.preprocess_features(val_data, is_training=False)
        
        # Create simple graph data for TGN
        print("\nCreating graph structure...")
        edges, edge_times, edge_features, node_map = self.create_simple_graph_data(sample_data)
        
        # Save processed data
        processed_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_columns': feature_cols,
            'train_ids': train_ids,
            'val_ids': val_ids,
            'train_times': train_times,
            'val_times': val_times,
            'edges': edges,
            'edge_timestamps': edge_times,
            'edge_features': edge_features,
            'node_mapping': node_map,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        with open(f"{self.data_path}/processed_data.pkl", 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"\n=== Preprocessing complete ===")
        print(f"Data saved to {self.data_path}/processed_data.pkl")
        print(f"Train: {X_train.shape}, Val: {X_val.shape}")
        
        return processed_data

if __name__ == "__main__":
    processor = FraudDataProcessor()
    data = processor.process_and_save()
