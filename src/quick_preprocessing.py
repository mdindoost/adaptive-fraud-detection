#!/usr/bin/env python3
"""
Quick preprocessing fix
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

class QuickProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def process_data(self):
        print("=== Quick Data Preprocessing ===")
        
        # Load data
        train_transaction = pd.read_csv("data/train_transaction.csv")
        train_identity = pd.read_csv("data/train_identity.csv")
        data = train_transaction.merge(train_identity, on='TransactionID', how='left')
        
        print(f"Loaded: {len(data)} transactions")
        
        # Sample for quick validation
        sample_data = data.sample(n=50000, random_state=42)
        
        # Temporal split
        sample_data = sample_data.sort_values('TransactionDT')
        split_point = int(0.8 * len(sample_data))
        train_data = sample_data.iloc[:split_point]
        val_data = sample_data.iloc[split_point:]
        
        print(f"Split: {len(train_data)} train, {len(val_data)} val")
        
        # Select simple features
        features = ['TransactionAmt', 'card1', 'card2', 'addr1', 'C1', 'C2', 'D1', 'D2']
        features = [f for f in features if f in train_data.columns]
        
        # Process training data
        X_train = train_data[features].fillna(-999)
        y_train = train_data['isFraud']
        train_times = train_data['TransactionDT']
        train_ids = train_data['TransactionID']
        
        # Process validation data (same features, same fillna value)
        X_val = val_data[features].fillna(-999)
        y_val = val_data['isFraud'] 
        val_times = val_data['TransactionDT']
        val_ids = val_data['TransactionID']
        
        # Scale features
        X_train_scaled = pd.DataFrame(self.scaler.fit_transform(X_train), 
                                     columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(self.scaler.transform(X_val),
                                   columns=X_val.columns, index=X_val.index)
        
        # Save processed data
        processed_data = {
            'X_train': X_train_scaled,
            'y_train': y_train,
            'X_val': X_val_scaled,
            'y_val': y_val,
            'feature_columns': features,
            'train_ids': train_ids,
            'val_ids': val_ids,
            'train_times': train_times,
            'val_times': val_times,
            'edges': np.array([]).reshape(0, 2),
            'edge_timestamps': np.array([]),
            'edge_features': np.array([]).reshape(0, 2),
            'node_mapping': {},
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }
        
        with open('data/processed_data.pkl', 'wb') as f:
            pickle.dump(processed_data, f)
        
        print("✅ Quick preprocessing complete!")
        print(f"Features: {features}")
        print(f"Train fraud rate: {y_train.mean():.4f}")
        print(f"Val fraud rate: {y_val.mean():.4f}")
        
        return processed_data

if __name__ == "__main__":
    processor = QuickProcessor()
    data = processor.process_data()
