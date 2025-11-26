import os
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class Predictor:
    def __init__(self):
        self.model_folder = Config.MODEL_FOLDER
        self.logistic_model = None
        self.rf_model = None
        self.cnn_model = None
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load all trained models"""
        # Load label encoder
        encoder_path = os.path.join(self.model_folder, 'label_encoder.joblib')
        if os.path.exists(encoder_path):
            self.label_encoder = joblib.load(encoder_path)
        
        # Load Logistic Regression
        lr_path = os.path.join(self.model_folder, 'logistic_regression.joblib')
        if os.path.exists(lr_path):
            self.logistic_model = joblib.load(lr_path)
        
        # Load Random Forest
        rf_path = os.path.join(self.model_folder, 'random_forest.joblib')
        if os.path.exists(rf_path):
            self.rf_model = joblib.load(rf_path)
        
        # Load CNN
        cnn_path = os.path.join(self.model_folder, 'cnn_model.h5')
        if os.path.exists(cnn_path):
            self.cnn_model = tf.keras.models.load_model(cnn_path)
    
    def preprocess_image(self, image_array):
        """Preprocess image for prediction"""
        # Resize and normalize
        image = Image.fromarray(image_array)
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        
        # Add batch dimension
        image_batch = np.expand_dims(image_array, axis=0)
        return image_batch
    
    def predict_logistic(self, image_array):
        """Predict using Logistic Regression"""
        if self.logistic_model is None or self.label_encoder is None:
            return None
        
        processed = self.preprocess_image(image_array)
        flattened = processed.reshape(1, -1)
        
        probabilities = self.logistic_model.predict_proba(flattened)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': float(probabilities[predicted_class_idx]),
            'all_probabilities': {
                cls: float(prob) for cls, prob in 
                zip(self.label_encoder.classes_, probabilities)
            }
        }
    
    def predict_random_forest(self, image_array):
        """Predict using Random Forest"""
        if self.rf_model is None or self.label_encoder is None:
            return None
        
        processed = self.preprocess_image(image_array)
        flattened = processed.reshape(1, -1)
        
        probabilities = self.rf_model.predict_proba(flattened)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': float(probabilities[predicted_class_idx]),
            'all_probabilities': {
                cls: float(prob) for cls, prob in 
                zip(self.label_encoder.classes_, probabilities)
            }
        }
    
    def predict_cnn(self, image_array):
        """Predict using CNN"""
        if self.cnn_model is None or self.label_encoder is None:
            return None
        
        processed = self.preprocess_image(image_array)
        
        probabilities = self.cnn_model.predict(processed)[0]
        predicted_class_idx = np.argmax(probabilities)
        predicted_class = self.label_encoder.classes_[predicted_class_idx]
        
        return {
            'class': predicted_class,
            'confidence': float(probabilities[predicted_class_idx]),
            'all_probabilities': {
                cls: float(prob) for cls, prob in 
                zip(self.label_encoder.classes_, probabilities)
            }
        }
    
    def predict_all_models(self, image_array):
        """Get predictions from all models"""
        predictions = {}
        
        predictions['logistic'] = self.predict_logistic(image_array)
        predictions['random_forest'] = self.predict_random_forest(image_array)
        predictions['cnn'] = self.predict_cnn(image_array)
        
        return predictions