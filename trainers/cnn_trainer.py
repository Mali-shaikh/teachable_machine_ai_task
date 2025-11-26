
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt

import os
import numpy as np
import tensorflow as tf
layers = tf.keras.layers
models = tf.keras.models
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class CNNTrainer:
    def __init__(self):
        self.model = None
        self.model_folder = Config.MODEL_FOLDER
        self.history = None
    
    def build_model(self, input_shape, num_classes):
        """Build CNN model architecture"""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X, y, label_encoder):
        """Train CNN model"""
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Build model
            self.model = self.build_model(X_train.shape[1:], len(label_encoder.classes_))
            
            # Train model with fewer epochs for testing
            self.history = self.model.fit(
                X_train, y_train,
                epochs=10,  # Reduced for faster training
                batch_size=32,
                validation_data=(X_test, y_test),
                verbose=0  # Suppress output
            )
            
            # Evaluate
            y_pred_proba = self.model.predict(X_test, verbose=0)
            y_pred = np.argmax(y_pred_proba, axis=1)
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            # Save model
            model_path = os.path.join(self.model_folder, 'cnn_model.h5')
            self.model.save(model_path)
            
            result = {
                'accuracy': float(accuracy),
                'confusion_matrix': cm.tolist(),
                'classes': label_encoder.classes_.tolist(),
                'model_type': 'CNN',
                'training_history': {
                    'accuracy': [float(x) for x in self.history.history['accuracy']],
                    'val_accuracy': [float(x) for x in self.history.history['val_accuracy']],
                    'loss': [float(x) for x in self.history.history['loss']],
                    'val_loss': [float(x) for x in self.history.history['val_loss']]
                }
            }
            
            return result
        except Exception as e:
            raise Exception(f"CNN training failed: {str(e)}")