from PIL import Image
import os
import streamlit as st

def validate_image_streamlit(file):
    """Validate uploaded image file for Streamlit"""
    if file is None:
        return False
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    file_extension = file.name.split('.')[-1].lower() if '.' in file.name else ''
    if file_extension not in allowed_extensions:
        return False
    
    # Validate image content
    try:
        image = Image.open(file)
        image.verify()
        file.seek(0)  # Reset stream position
        return True
    except Exception:
        return False


def validate_image_streamlit(file):
    """Validate uploaded image file for Streamlit"""
    if file is None:
        return False
    
    # Check file extension
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if hasattr(file, 'name'):
        file_extension = file.name.split('.')[-1].lower() if '.' in file.name else ''
    else:
        return False
        
    if file_extension not in allowed_extensions:
        return False
    
    # Validate image content
    try:
        from PIL import Image
        image = Image.open(file)
        image.verify()
        file.seek(0)  # Reset stream position for Streamlit files
        return True
    except Exception:
        return False