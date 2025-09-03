#!/usr/bin/env python3
"""
Statistical baseline models for fraud detection comparison
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

class StatisticalBaselines:
    def __init__(self):
        self.models = {}
        self.training_times = {}
        self.inference_times = {}
        
    def train_logistic_regression(self, X_train, y_train):
        """Train logistic regression with class balancing"""
        print("Training Logistic Regression...")
        
        start_time = time.time()
        
        # Use balanced class weights for imbalanced data
        lr = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42,
            n_jobs=-1
        )
        
        lr.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.models['logistic_regression'] = lr
        self.training_times['logistic_regression'] = training_time
        
        print(f"Logistic Regression trained in {training_time:.2f} seconds")
        return lr
    
    def train_random_forest(self, X_train, y_train):
        """Train random forest with class balancing"""
        print("Training Random Forest...")
        
        start_time = time.time()
        
        # Use balanced class weights and limit trees for faster inference
        rf = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        rf.fit(X_train, y_train)
        
        training_time = time.time() - start_time
        self.models['random_forest'] = rf
        self.training_times['random_forest'] = training_time
        
        print(f"Random Forest trained in {training_time:.2f} seconds")
        return rf
    
    def evaluate_model(self, model_name, X_val, y_val, batch_size=1000):
        """Evaluate model performance and measure inference time"""
        model = self.models[model_name]
        
        # Measure inference time on batches (simulating real-time processing)
        inference_times = []
        predictions = []
        probabilities = []
        
        for i in range(0, len(X_val), batch_size):
            batch_X = X_val.iloc[i:i+batch_size]
            
            # Measure inference time
            start_time = time.time()
            batch_pred = model.predict(batch_X)
            batch_prob = model.predict_proba(batch_X)[:, 1]  # Probability of fraud
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            predictions.extend(batch_pred)
            probabilities.extend(batch_prob)
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        auc_roc = roc_auc_score(y_val, probabilities)
        precision, recall, _ = precision_recall_curve(y_val, probabilities)
        auc_pr = auc(recall, precision)
        
        # Average inference time per transaction
        avg_inference_time = np.mean(inference_times) / batch_size * 1000  # milliseconds per transaction
        self.inference_times[model_name] = avg_inference_time
        
        results = {
            'model': model_name,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'training_time': self.training_times[model_name],
            'inference_time_ms': avg_inference_time,
            'predictions': predictions,
            'probabilities': probabilities
        }
        
        print(f"\n{model_name.upper()} Results:")
        print(f"AUC-ROC: {auc_roc:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"Training time: {self.training_times[model_name]:.2f} seconds")
        print(f"Inference time: {avg_inference_time:.2f} ms per transaction")
        
        return results
    
    def train_all_models(self, X_train, y_train):
        """Train all statistical baseline models"""
        print("=== Training Statistical Baselines ===\n")
        
        self.train_logistic_regression(X_train, y_train)
        self.train_random_forest(X_train, y_train)
        
        print(f"\nAll models trained successfully!")
        
    def evaluate_all_models(self, X_val, y_val):
        """Evaluate all trained models"""
        print("\n=== Evaluating Statistical Baselines ===")
        
        results = {}
        for model_name in self.models.keys():
            results[model_name] = self.evaluate_model(model_name, X_val, y_val)
        
        return results
    
    def get_feature_importance(self, model_name, feature_names):
        """Get feature importance for interpretability"""
        if model_name not in self.models:
            return None
            
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            # Random Forest
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # Logistic Regression
            importance = np.abs(model.coef_[0])
        else:
            return None
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def save_models(self, filepath):
        """Save all trained models"""
        model_data = {
            'models': self.models,
            'training_times': self.training_times,
            'inference_times': self.inference_times
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Models saved to {filepath}")

if __name__ == "__main__":
    # Load processed data
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    feature_columns = data['feature_columns']
    
    # Train and evaluate statistical baselines
    baselines = StatisticalBaselines()
    baselines.train_all_models(X_train, y_train)
    results = baselines.evaluate_all_models(X_val, y_val)
    
    # Show feature importance
    print("\n=== Feature Importance (Random Forest) ===")
    importance = baselines.get_feature_importance('random_forest', feature_columns)
    print(importance.head(10))
    
    # Save models
    baselines.save_models('results/statistical_models.pkl')
