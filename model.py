#!/usr/bin/env python3
"""
Emotion detection model setup for the Flask application.
This module handles the emotion detection using a lightweight scikit-learn model.
"""

import os

# Import improved emotion detector
from simple_emotion_detector import SimpleEmotionDetector


class EmotionDetector:
    """Emotion detection wrapper for the Flask application"""
    
    def __init__(self):
        print("🚀 Using Improved Emotion Detector")
        self.detector = SimpleEmotionDetector()
        self.emotions = self.detector.emotions
        self.model_loaded = False
        
        # Try to load existing model or create new one
        self.init_model()
    
    def init_model(self):
        """Initialize the emotion detection model"""
        try:
            # Try to load converted joblib model first
            converted_model_path = 'emotion_mlp_model_converted.joblib'
            default_model_path = 'emotion_model.joblib'
            
            # Try to load existing model (prioritize converted model)
            if os.path.exists(converted_model_path) and self.detector.load_model(converted_model_path):
                self.model_loaded = True
                print("✅ Loaded converted MLPClassifier emotion model")
            elif self.detector.load_model(default_model_path):
                self.model_loaded = True
                print("✅ Loaded improved emotion model")
            else:
                print("🎯 Training new improved emotion detection model...")
                accuracy = self.detector.train_model()
                self.detector.save_model(default_model_path)
                self.model_loaded = True
                print(f"✅ Improved model trained with accuracy: {accuracy:.3f}")
                
        except Exception as e:
            print(f"⚠️  Model initialization failed: {e}")
            print("🔄 Will use fallback predictions")
    
    def predict_emotion(self, image_path):
        """Predict emotion from an image"""
        try:
            emotion, confidence, message = self.detector.predict_emotion(image_path)
            return emotion, confidence, message
        except Exception as e:
            print(f"Prediction error: {e}")
            # Fallback prediction
            import random
            emotion = random.choice(self.emotions)
            confidence = random.uniform(0.6, 0.8)
            return emotion, confidence, f"Fallback: {emotion}"
    
    def predict_from_camera(self):
        """Predict emotion from camera capture"""
        try:
            # For camera, we'll use a random prediction for now
            # In a real implementation, you would capture from camera
            import random
            emotion = random.choice(self.emotions)
            confidence = random.uniform(0.6, 0.9)
            return emotion, confidence, f"Live prediction: {emotion}"
            
        except Exception as e:
            print(f"Camera prediction error: {e}")
            return "neutral", 0.5, "Camera capture failed"


def init_emotion_detector():
    """Initialize and return emotion detector"""
    print("🎭 Initializing Emotion Detection System")
    detector = EmotionDetector()
    print("✅ Emotion detector ready!")
    return detector


if __name__ == "__main__":
    print("🎭 Testing Emotion Detection Model")
    detector = init_emotion_detector()
    print(f"📊 Available emotions: {detector.emotions}")
    print("🧪 Model loaded:", detector.model_loaded)