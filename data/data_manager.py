import os
import json
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime
import sys
import streamlit as st

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

class DataManager:
    def __init__(self):
        self.dataset_folder = Config.DATASET_FOLDER
        self.model_folder = Config.MODEL_FOLDER
        
    def create_class(self, class_name):
        """Create a new class directory"""
        class_path = os.path.join(self.dataset_folder, class_name)
        if os.path.exists(class_path):
            return False
        
        os.makedirs(class_path, exist_ok=True)
        return True
    
    def save_image(self, file, class_name):
        """Save uploaded image to class directory - Streamlit compatible"""
        class_path = os.path.join(self.dataset_folder, class_name)
        if not os.path.exists(class_path):
            return False
        
        # Generate unique filename - handle both Flask and Streamlit file objects
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Handle different file object types
        if hasattr(file, 'filename'):  # Flask file object
            original_filename = file.filename
        elif hasattr(file, 'name'):    # Streamlit file object
            original_filename = file.name
        else:
            original_filename = "image.jpg"
        
        # Clean filename and ensure proper extension
        import re
        clean_name = re.sub(r'[^\w\.-]', '_', original_filename)
        filename = f"{timestamp}_{clean_name}"
        filepath = os.path.join(class_path, filename)
        
        try:
            # Handle both Flask and Streamlit file objects
            if hasattr(file, 'save'):  # Flask file object
                file.save(filepath)
            else:  # Streamlit file object - read bytes and save
                with open(filepath, 'wb') as f:
                    f.write(file.getvalue())
            
            # Verify the image was saved correctly
            if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                return True
            else:
                # Clean up if file wasn't saved properly
                if os.path.exists(filepath):
                    os.remove(filepath)
                return False
                
        except Exception as e:
            print(f"Error saving image: {e}")
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return False
    
    def load_training_data(self, class_names):
        """Load images and labels for training"""
        images = []
        labels = []
        
        for class_name in class_names:
            class_path = os.path.join(self.dataset_folder, class_name)
            if not os.path.exists(class_path):
                st.warning(f"Class directory not found: {class_name}")
                continue
        
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
            if not image_files:
                st.warning(f"No images found in class: {class_name}")
                continue
            
            for filename in image_files:
                filepath = os.path.join(class_path, filename)
                try:
                    image = Image.open(filepath)
                    image = image.convert('RGB')
                    image = image.resize((64, 64))  # Resize for consistency
                    image_array = np.array(image) / 255.0  # Normalize
                    images.append(image_array)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {filepath}: {e}")
    
        if len(images) == 0:
            st.error("No valid images found for training!")
            return np.array([]), np.array([]), None
    
        X = np.array(images)
        y = np.array(labels)
    
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    
        return X, y_encoded, label_encoder
    
    def get_all_classes(self):
        """Get list of all created classes"""
        if not os.path.exists(self.dataset_folder):
            return []
        return [d for d in os.listdir(self.dataset_folder) 
                if os.path.isdir(os.path.join(self.dataset_folder, d))]
    
    def get_image_count(self, class_name):
        """Get number of images in a class"""
        class_path = os.path.join(self.dataset_folder, class_name)
        if not os.path.exists(class_path):
            return 0
        return len([f for f in os.listdir(class_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    def save_label_encoder(self, label_encoder):
        """Save label encoder for inference"""
        encoder_path = os.path.join(self.model_folder, 'label_encoder.joblib')
        joblib.dump(label_encoder, encoder_path)
    
    def load_label_encoder(self):
        """Load label encoder for inference"""
        encoder_path = os.path.join(self.model_folder, 'label_encoder.joblib')
        if os.path.exists(encoder_path):
            return joblib.load(encoder_path)
        return None
    
    def save_training_result(self, model_name, result):
        """Save training results to history"""
        history_path = os.path.join(self.model_folder, 'training_history.json')
        history = self.load_training_history()
        
        result['timestamp'] = datetime.now().isoformat()
        result['model_name'] = model_name
        
        history.append(result)
        
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
    
    def load_training_history(self):
        """Load training history"""
        history_path = os.path.join(self.model_folder, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                return json.load(f)
        return []
    
    def clear_all_data(self):
        """Clear all datasets and models"""
        import shutil
        if os.path.exists(self.dataset_folder):
            shutil.rmtree(self.dataset_folder)
            os.makedirs(self.dataset_folder)
        if os.path.exists(self.model_folder):
            shutil.rmtree(self.model_folder)
            os.makedirs(self.model_folder)
    
    def get_sample_images(self, class_name, max_samples=3):
        """Get sample image paths from a class"""
        class_path = os.path.join(self.dataset_folder, class_name)
        if not os.path.exists(class_path):
            return []
        
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        sample_files = image_files[:max_samples]
        return [os.path.join(class_path, f) for f in sample_files]
# Add this at the very end of the file
if __name__ == "__main__":
    print("✅ data_manager.py loaded successfully - no syntax errors!")