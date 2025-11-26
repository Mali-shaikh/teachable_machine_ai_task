import os
import numpy as np
import joblib
from PIL import Image
import sys

# Conditional TensorFlow import with error handling
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError as e:
    print(f"Warning: TensorFlow not available: {e}")
    tf = None
    TF_AVAILABLE = False

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
        print(f"[INFO] Loading models from: {self.model_folder}")
        print(f"[INFO] Model folder exists: {os.path.exists(self.model_folder)}")
        
        if os.path.exists(self.model_folder):
            print(f"[INFO] Files in model folder: {os.listdir(self.model_folder)}")
        
        # Load label encoder
        encoder_path = os.path.join(self.model_folder, 'label_encoder.joblib')
        if os.path.exists(encoder_path):
            try:
                self.label_encoder = joblib.load(encoder_path)
                print(f"✓ Label encoder loaded successfully")
                print(f"  Classes: {self.label_encoder.classes_}")
            except Exception as e:
                print(f"✗ Error loading label encoder: {e}")
        else:
            print(f"✗ Label encoder not found at: {encoder_path}")
        
        # Load Logistic Regression
        lr_path = os.path.join(self.model_folder, 'logistic_regression.joblib')
        if os.path.exists(lr_path):
            try:
                self.logistic_model = joblib.load(lr_path)
                print(f"✓ Logistic Regression loaded successfully")
            except Exception as e:
                print(f"✗ Error loading Logistic Regression: {e}")
        else:
            print(f"✗ Logistic Regression not found at: {lr_path}")
        
        # Load Random Forest
        rf_path = os.path.join(self.model_folder, 'random_forest.joblib')
        if os.path.exists(rf_path):
            try:
                self.rf_model = joblib.load(rf_path)
                print(f"✓ Random Forest loaded successfully")
            except Exception as e:
                print(f"✗ Error loading Random Forest: {e}")
        else:
            print(f"✗ Random Forest not found at: {rf_path}")
        
        # Load CNN with enhanced error handling
        if not TF_AVAILABLE:
            print(f"✗ TensorFlow not available - CNN model cannot be loaded")
            self.cnn_model = None
            return
        
        cnn_path = os.path.join(self.model_folder, 'cnn_model.h5')
        print(f"[INFO] Looking for CNN model at: {cnn_path}")
        print(f"[INFO] CNN file exists: {os.path.exists(cnn_path)}")
        
        if os.path.exists(cnn_path):
            try:
                print(f"[INFO] Attempting to load CNN model...")
                print(f"[INFO] TensorFlow version: {tf.__version__}")
                
                # Load without compiling for maximum compatibility
                self.cnn_model = tf.keras.models.load_model(
                    cnn_path, 
                    compile=False
                )
                
                print(f"[INFO] CNN model loaded, now recompiling...")
                
                # Recompile with explicit optimizer settings
                self.cnn_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )
                
                print(f"✓ CNN model loaded and compiled successfully")
                print(f"  Input shape: {self.cnn_model.input_shape}")
                print(f"  Output shape: {self.cnn_model.output_shape}")
                
            except Exception as e:
                print(f"✗ Error loading CNN model: {str(e)}")
                print(f"  Error type: {type(e).__name__}")
                import traceback
                print(f"  Traceback:\n{traceback.format_exc()}")
                self.cnn_model = None
        else:
            print(f"✗ CNN model file not found at: {cnn_path}")
            self.cnn_model = None
    
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
            print("[WARN] Logistic model or label encoder not available")
            return None
        
        try:
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
        except Exception as e:
            print(f"[ERROR] Logistic prediction failed: {e}")
            return None
    
    def predict_random_forest(self, image_array):
        """Predict using Random Forest"""
        if self.rf_model is None or self.label_encoder is None:
            print("[WARN] Random Forest model or label encoder not available")
            return None
        
        try:
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
        except Exception as e:
            print(f"[ERROR] Random Forest prediction failed: {e}")
            return None
    
    def predict_cnn(self, image_array):
        """Predict using CNN"""
        if self.cnn_model is None or self.label_encoder is None:
            print("[WARN] CNN model or label encoder not available")
            return None
        
        try:
            processed = self.preprocess_image(image_array)
            
            probabilities = self.cnn_model.predict(processed, verbose=0)[0]
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
        except Exception as e:
            print(f"[ERROR] CNN prediction failed: {e}")
            return None
    
    def predict_all_models(self, image_array):
        """Get predictions from all models"""
        predictions = {}
        
        predictions['logistic'] = self.predict_logistic(image_array)
        predictions['random_forest'] = self.predict_random_forest(image_array)
        predictions['cnn'] = self.predict_cnn(image_array)
        
        return predictions