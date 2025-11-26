import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class LogisticTrainer:
    def __init__(self):
        self.model = None
        self.model_folder = Config.MODEL_FOLDER
    
    def train(self, X, y, label_encoder):
        """Train Logistic Regression model"""
        try:
            # Flatten images for traditional ML models
            X_flat = X.reshape(X.shape[0], -1)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_flat, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train model
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                multi_class='multinomial'
            )
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Save model
            model_path = os.path.join(self.model_folder, 'logistic_regression.joblib')
            joblib.dump(self.model, model_path)
            
            result = {
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist(),
                'classes': label_encoder.classes_.tolist(),
                'model_type': 'Logistic Regression'
            }
            
            return result
        except Exception as e:
            raise Exception(f"Logistic Regression training failed: {str(e)}")