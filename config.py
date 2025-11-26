import os

class Config:
    SECRET_KEY = 'your-secret-key-here'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_uploads')
    MODEL_FOLDER = os.path.join(BASE_DIR, 'models', 'saved_models')
    DATASET_FOLDER = os.path.join(BASE_DIR, 'data', 'datasets')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    MIN_IMAGES_PER_CLASS = 10
    
    @staticmethod
    def init_app(app):
        # Create necessary directories
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.MODEL_FOLDER, exist_ok=True)
        os.makedirs(Config.DATASET_FOLDER, exist_ok=True)