import streamlit as st
import os
import json
import base64
import io
from datetime import datetime
import numpy as np
from PIL import Image

# Set matplotlib backend to Agg for Streamlit Cloud compatibility
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import sys

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your custom modules
from data.data_manager import DataManager
from trainers.logistic_trainer import LogisticTrainer
from trainers.random_forest_trainer import RandomForestTrainer
from trainers.cnn_trainer import CNNTrainer
from inference.predictor import Predictor
from utils.validators import validate_image_streamlit

# Import config directly
from config import Config

# Page configuration
st.set_page_config(
    page_title="AI Teachable Machine",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
data_manager = DataManager()
predictor = Predictor()

def main():
    st.title("🤖 AI Teachable Machine")
    st.markdown("Create custom image classifiers with multiple ML models")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox(
        "Choose App Mode",
        ["🏠 Home", "📁 Manage Classes", "🖼️ Upload Images", "🧠 Train Models", "🔍 Predict", "📊 Results"]
    )
    
    if app_mode == "🏠 Home":
        show_home()
    elif app_mode == "📁 Manage Classes":
        manage_classes()
    elif app_mode == "🖼️ Upload Images":
        upload_images()
    elif app_mode == "🧠 Train Models":
        train_models()
    elif app_mode == "🔍 Predict":
        predict_images()
    elif app_mode == "📊 Results":
        show_results()

def show_home():
    st.header("Welcome to AI Teachable Machine")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🚀 Features")
        st.markdown("""
        - **Multi-class Image Classification**
        - **Multiple ML Models**:
          - Logistic Regression
          - Random Forest  
          - CNN (TensorFlow)
        - **Real-time Predictions**
        - **Webcam Support**
        - **Model Comparison**
        """)
        
        st.subheader("📋 Quick Start")
        st.markdown("""
        1. **Create Classes** - Define your categories
        2. **Upload Images** - Add training images
        3. **Train Models** - Train multiple ML models
        4. **Predict** - Test with images or webcam
        """)
    
    with col2:
        st.subheader("📊 Current Status")
        
        # Show current classes and image counts
        classes = data_manager.get_all_classes()
        if classes:
            st.write("**Existing Classes:**")
            for class_name in classes:
                count = data_manager.get_image_count(class_name)
                st.write(f"- {class_name}: {count} images")
        else:
            st.info("No classes created yet. Go to 'Manage Classes' to get started!")
        
        # Show model status
        st.write("**Model Status:**")
        predictor.load_models()
        models_loaded = {
            "Logistic Regression": predictor.logistic_model is not None,
            "Random Forest": predictor.rf_model is not None,
            "CNN": predictor.cnn_model is not None
        }
        
        for model_name, loaded in models_loaded.items():
            status = "✅ Loaded" if loaded else "❌ Not trained"
            st.write(f"- {model_name}: {status}")

def manage_classes():
    st.header("📁 Manage Classes")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Create New Class")
        with st.form("create_class_form"):
            class_name = st.text_input("Class Name", placeholder="e.g., Cats, Dogs, Cars...")
            submit_button = st.form_submit_button("Create Class")
            
            if submit_button and class_name:
                if data_manager.create_class(class_name):
                    st.success(f"Class '{class_name}' created successfully!")
                    st.rerun()
                else:
                    st.error(f"Class '{class_name}' already exists!")
    
    with col2:
        st.subheader("Existing Classes")
        classes = data_manager.get_all_classes()
        
        if not classes:
            st.info("No classes created yet.")
        else:
            for class_name in classes:
                count = data_manager.get_image_count(class_name)
                col_a, col_b, col_c = st.columns([3, 1, 1])
                with col_a:
                    st.write(f"**{class_name}**")
                with col_b:
                    st.write(f"{count} images")
                with col_c:
                    if st.button("Delete", key=f"delete_{class_name}"):
                        # Simple delete implementation
                        class_path = os.path.join(Config.DATASET_FOLDER, class_name)
                        if os.path.exists(class_path):
                            import shutil
                            shutil.rmtree(class_path)
                            st.success(f"Class '{class_name}' deleted!")
                            st.rerun()
            
            # Clear all data
            if st.button("🗑️ Clear All Data", type="secondary"):
                if st.checkbox("I understand this will delete ALL classes and images"):
                    data_manager.clear_all_data()
                    st.success("All data cleared successfully!")
                    st.rerun()

def upload_images():
    st.header("🖼️ Upload Images")
    
    classes = data_manager.get_all_classes()
    
    if not classes:
        st.warning("Please create classes first in the 'Manage Classes' section.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Images")
        selected_class = st.selectbox("Select Class", classes)
        
        uploaded_files = st.file_uploader(
            "Choose images",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            help="Select multiple images for the chosen class"
        )
        
        if uploaded_files and selected_class:
            if st.button("Upload Images"):
                valid_count = 0
                progress_bar = st.progress(0)
                status_text = st.empty()
                error_messages = []
                
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing image {i+1}/{len(uploaded_files)}: {file.name}")
                    
                    # Validate image
                    if validate_image_streamlit(file):
                        # Save image
                        if data_manager.save_image(file, selected_class):
                            valid_count += 1
                        else:
                            error_messages.append(f"Failed to save: {file.name}")
                    else:
                        error_messages.append(f"Invalid image: {file.name}")
                    
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                status_text.text("")
                
                # Show results
                if valid_count > 0:
                    st.success(f"✅ Successfully uploaded {valid_count} out of {len(uploaded_files)} images!")
                
                if error_messages:
                    with st.expander("Show error details"):
                        for error in error_messages[:5]:  # Show first 5 errors
                            st.error(error)
                        if len(error_messages) > 5:
                            st.info(f"... and {len(error_messages) - 5} more errors")
                
                st.rerun()
    
    with col2:
        st.subheader("Class Overview")
        for class_name in classes:
            count = data_manager.get_image_count(class_name)
            st.write(f"**{class_name}**: {count} images")
            
            # Show sample images if available
            if count > 0:
                sample_images = data_manager.get_sample_images(class_name, max_samples=3)
                if sample_images:
                    cols = st.columns(len(sample_images))
                    for col, img_path in zip(cols, sample_images):
                        try:
                            image = Image.open(img_path)
                            col.image(image, caption=class_name)
                        except Exception as e:
                            st.error(f"Error loading image: {e}")

def train_models():
    st.header("🧠 Train Models")
    
    classes = data_manager.get_all_classes()
    
    if not classes:
        st.warning("Please create classes and upload images first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Setup")
        
        selected_classes = st.multiselect(
            "Select Classes to Train On",
            classes,
            default=classes
        )
        
        st.subheader("Select Models")
        train_logistic = st.checkbox("Logistic Regression", value=True)
        train_rf = st.checkbox("Random Forest", value=True)
        train_cnn = st.checkbox("CNN (TensorFlow)", value=True)
        
        model_types = []
        if train_logistic:
            model_types.append('logistic')
        if train_rf:
            model_types.append('random_forest')
        if train_cnn:
            model_types.append('cnn')
        
        # Check if we have enough images
        total_images = sum(data_manager.get_image_count(cls) for cls in selected_classes)
        min_required = Config.MIN_IMAGES_PER_CLASS * len(selected_classes)
        
        st.info(f"Total images: {total_images} | Minimum required: {min_required}")
        
        if total_images < min_required:
            st.error(f"Need at least {Config.MIN_IMAGES_PER_CLASS} images per class. Current: {total_images}/{min_required}")
        else:
            st.success("✓ Sufficient images available for training!")
    
    with col2:
        st.subheader("Start Training")
        
        if st.button("🚀 Train Models", disabled=total_images < min_required or not model_types or not selected_classes):
            if not selected_classes:
                st.error("Please select at least one class.")
                return
            
            if not model_types:
                st.error("Please select at least one model.")
                return
            
            # Training progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            results_container = st.empty()
            
            try:
                # Load training data
                status_text.text("📥 Loading training data...")
                X, y, label_encoder = data_manager.load_training_data(selected_classes)
                
                if X.shape[0] == 0:
                    st.error("No valid training data found. Please check your images.")
                    return
                
                progress_bar.progress(20)
                status_text.text(f"📊 Loaded {X.shape[0]} images for training")
                
                results = {}
                current_progress = 20
                progress_step = 60 / len(model_types)  # Distribute remaining progress
                
                # Train models
                if 'logistic' in model_types:
                    status_text.text("🤖 Training Logistic Regression...")
                    try:
                        logistic_trainer = LogisticTrainer()
                        results['logistic'] = logistic_trainer.train(X, y, label_encoder)
                        data_manager.save_training_result('logistic', results['logistic'])
                        st.success("✅ Logistic Regression trained successfully!")
                    except Exception as e:
                        st.error(f"❌ Logistic Regression failed: {str(e)}")
                        results['logistic'] = None
                    
                    current_progress += progress_step
                    progress_bar.progress(int(current_progress))
                
                if 'random_forest' in model_types:
                    status_text.text("🌲 Training Random Forest...")
                    try:
                        rf_trainer = RandomForestTrainer()
                        results['random_forest'] = rf_trainer.train(X, y, label_encoder)
                        data_manager.save_training_result('random_forest', results['random_forest'])
                        st.success("✅ Random Forest trained successfully!")
                    except Exception as e:
                        st.error(f"❌ Random Forest failed: {str(e)}")
                        results['random_forest'] = None
                    
                    current_progress += progress_step
                    progress_bar.progress(int(current_progress))
                
                if 'cnn' in model_types:
                    status_text.text("🕸️ Training CNN...")
                    try:
                        cnn_trainer = CNNTrainer()
                        results['cnn'] = cnn_trainer.train(X, y, label_encoder)
                        data_manager.save_training_result('cnn', results['cnn'])
                        st.success("✅ CNN trained successfully!")
                    except Exception as e:
                        st.error(f"❌ CNN failed: {str(e)}")
                        results['cnn'] = None
                
                # Save label encoder
                data_manager.save_label_encoder(label_encoder)
                progress_bar.progress(95)
                
                # Reload models for prediction
                predictor.load_models()
                progress_bar.progress(100)
                status_text.text("🎉 Training completed!")
                
                # Display results
                st.balloons()
                display_training_results(results)
                
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def display_training_results(results):
    st.subheader("📊 Training Results")
    
    for model_name, result in results.items():
        if result:  # Only show if result exists
            with st.expander(f"{result['model_type']} - Accuracy: {result['accuracy']*100:.2f}%"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Metrics:**")
                    st.metric("Accuracy", f"{result['accuracy']*100:.2f}%")
                    st.write(f"**Classes:** {', '.join(result['classes'])}")
                
                with col2:
                    st.write("**Confusion Matrix:**")
                    # Create a simple visualization of confusion matrix
                    cm = np.array(result['confusion_matrix'])
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                               xticklabels=result['classes'], 
                               yticklabels=result['classes'],
                               ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                
                # Show training history for CNN
                if 'training_history' in result:
                    st.write("**Training History:**")
                    history = result['training_history']
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Accuracy plot
                    ax1.plot(history['accuracy'], label='Training Accuracy')
                    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
                    ax1.set_title('Model Accuracy')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Accuracy')
                    ax1.legend()
                    
                    # Loss plot
                    ax2.plot(history['loss'], label='Training Loss')
                    ax2.plot(history['val_loss'], label='Validation Loss')
                    ax2.set_title('Model Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Loss')
                    ax2.legend()
                    
                    st.pyplot(fig)

def predict_images():
    st.header("🔍 Make Predictions")
    
    # Check if models are trained
    predictor.load_models()
    if predictor.label_encoder is None:
        st.warning("Please train models first in the 'Train Models' section.")
        return
    
    st.subheader("Upload Image for Classification")
    uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                image_array = np.array(image)
                predictions = predictor.predict_all_models(image_array)
                display_predictions(predictions)

def display_predictions(predictions):
    st.subheader("🤖 Model Predictions")
    
    for model_name, prediction in predictions.items():
        if prediction:
            st.markdown(f"**{format_model_name(model_name)}**")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                confidence_percent = prediction['confidence'] * 100
                st.metric(
                    "Prediction", 
                    prediction['class'],
                    delta=f"{confidence_percent:.1f}% confidence"
                )
            
            with col2:
                # Confidence bar
                st.progress(confidence_percent / 100)
                
                # All probabilities
                st.write("**All probabilities:**")
                for cls, prob in sorted(prediction['all_probabilities'].items(), 
                                      key=lambda x: x[1], reverse=True):
                    prob_percent = prob * 100
                    st.write(f"{cls}: {prob_percent:.1f}%")
            
            st.markdown("---")

def show_results():
    st.header("📊 Training Results & History")
    
    history = data_manager.load_training_history()
    
    if not history:
        st.info("No training history yet. Train some models first!")
        return
    
    # Show latest training results
    st.subheader("Latest Training Session")
    latest = history[-1] if history else None
    
    if latest:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", latest.get('model_type', 'Unknown'))
        with col2:
            st.metric("Accuracy", f"{latest.get('accuracy', 0)*100:.2f}%")
        with col3:
            st.metric("Classes", len(latest.get('classes', [])))
    
    # Show all training history
    st.subheader("All Training Sessions")
    
    if history:
        # Create a dataframe for better display
        history_data = []
        for session in history:
            history_data.append({
                'Model': session.get('model_type', 'Unknown'),
                'Accuracy': f"{session.get('accuracy', 0)*100:.2f}%",
                'Classes': ', '.join(session.get('classes', [])),
                'Date': session.get('timestamp', 'Unknown')
            })
        
        df = pd.DataFrame(history_data)
        st.dataframe(df, use_container_width=True)

def format_model_name(model_name):
    names = {
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest', 
        'cnn': 'Convolutional Neural Network'
    }
    return names.get(model_name, model_name)

if __name__ == "__main__":
    main()