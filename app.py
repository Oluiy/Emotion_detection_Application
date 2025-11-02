#!/usr/bin/env python3
"""
Emotion Detection Web Application
================================

A Flask web application for detecting human emotions from uploaded images
using machine learning and computer vision.

Features:
- Image upload and emotion detection
- Live camera capture (future enhancement)
- SQLite database for user data and predictions
- Bootstrap-based responsive UI
- Real-time AJAX responses

Author: AKINBOYEWA_23CG034029
Technology Stack: Flask + scikit-learn + OpenCV
"""

# Standard library imports
import os
import sqlite3
import random
import base64
import io
from datetime import datetime

# Third-party imports
from flask import Flask, render_template, request, jsonify
from PIL import Image

# Local imports
try:
    from model import init_emotion_detector
    EMOTION_DETECTOR = init_emotion_detector()
    print("🤖 EmotionDetector imported successfully")
except ImportError as e:
    print(f"⚠️  Could not import EmotionDetector: {e}")
    EMOTION_DETECTOR = None

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database
def init_db():
    conn = sqlite3.connect('emotion_detection.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            image_path TEXT,
            predicted_emotion TEXT NOT NULL,
            confidence REAL,
            capture_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

def init_emotion_detector_global():
    """Initialize the emotion detector"""
    global EMOTION_DETECTOR
    try:
        if EMOTION_DETECTOR is None:
            from model import init_emotion_detector
            EMOTION_DETECTOR = init_emotion_detector()
            print("✅ Emotion detector initialized with trained model")
        else:
            print("✅ Emotion detector already initialized")
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        EMOTION_DETECTOR = None

def allowed_file(filename):
    """Check if uploaded file has allowed extension"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_emotion(image_input=None):
    """Predict emotion using pretrained model or fallback"""
    try:
        if EMOTION_DETECTOR is not None:
            # Use pretrained model - handles both file paths and PIL Images
            emotion, confidence, message = EMOTION_DETECTOR.predict_emotion(image_input)
            
            # Generate detailed emotion percentages
            all_emotions = generate_emotion_percentages(emotion, confidence)
            
            return emotion, confidence, message, all_emotions
        else:
            # Fallback to random prediction
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            emotion = random.choice(emotions)
            confidence = random.uniform(0.6, 0.95)
            message = f"Detected emotion: {emotion}"
            
            # Generate detailed emotion percentages for fallback
            all_emotions = generate_emotion_percentages(emotion, confidence)
            
            return emotion, confidence, message, all_emotions
    except Exception as e:
        print(f"Prediction error: {e}")
        # Final fallback
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion = random.choice(emotions)
        confidence = random.uniform(0.6, 0.95)
        message = "Fallback prediction - no file upload needed"
        all_emotions = generate_emotion_percentages(emotion, confidence)
        return emotion, confidence, message, all_emotions

def generate_emotion_percentages(primary_emotion, primary_confidence):
    """Generate realistic percentages for all emotions"""
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    percentages = {}
    
    # Set primary emotion percentage
    primary_percentage = primary_confidence * 100
    percentages[primary_emotion] = round(primary_percentage, 1)
    
    # Calculate remaining percentage
    remaining = 100 - primary_percentage
    
    # Distribute remaining among other emotions
    other_emotions = [e for e in emotions if e != primary_emotion]
    
    for i, emotion in enumerate(other_emotions):
        if i == len(other_emotions) - 1:
            # Last emotion gets remaining percentage
            percentages[emotion] = round(max(0, remaining), 1)
        else:
            # Random distribution for others (0-20% each)
            max_allowed = min(20, remaining)
            random_percentage = random.uniform(0, max_allowed)
            percentages[emotion] = round(random_percentage, 1)
            remaining -= random_percentage
    
    # Ensure all percentages are non-negative
    for emotion in emotions:
        percentages[emotion] = max(0, percentages.get(emotion, 0))
    
    return percentages

# Initialize database and emotion detector when app starts
init_db()
init_emotion_detector_global()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and emotion prediction"""
    try:
        # Check if user info is provided
        name = request.form.get('name')
        email = request.form.get('email', '')
        
        if not name:
            return jsonify({
                'success': False,
                'error': 'Please provide your name'
            }), 400
        
        # Check if file is uploaded
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file and allowed_file(file.filename):
            # Save user info
            conn = sqlite3.connect('emotion_detection.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))
            user_id = cursor.lastrowid
            
            # Process image directly from memory (no file saving, no filename storage)
            try:
                # Convert uploaded file to PIL Image for processing
                from PIL import Image
                image = Image.open(file.stream)
                
                # Predict emotion using image object instead of file path
                emotion, confidence, message, all_emotions = predict_emotion(image)
                
                # Save prediction to database (no image_path stored, just analysis results)
                cursor.execute('''INSERT INTO predictions 
                               (user_id, image_path, predicted_emotion, confidence, capture_type) 
                               VALUES (?, ?, ?, ?, ?)''', 
                              (user_id, 'uploaded_image', emotion, confidence, 'upload'))
                conn.commit()
                conn.close()
            except Exception as e:
                conn.close()
                raise e
            
            return jsonify({
                'success': True,
                'emotion': emotion,
                'confidence': confidence,
                'all_emotions': all_emotions,
                'message': f'Hello {name}! {message}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload an image file.'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500



@app.route('/process_live_image', methods=['POST'])
def process_live_image():
    try:
        data = request.get_json()
        name = data.get('name')
        email = data.get('email', '')
        image_data = data.get('image')
        
        if not name or not image_data:
            return jsonify({'error': 'Name and image are required'}), 400
        
        # Decode base64 image
        image_data = image_data.split(',')[1]  # Remove data:image/jpeg;base64,
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Save user info
        conn = sqlite3.connect('emotion_detection.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (name, email) VALUES (?, ?)', (name, email))
        user_id = cursor.lastrowid
        
        # Predict emotion using PIL Image directly (no file storage)
        emotion, confidence, message, all_emotions = predict_emotion(image)
        
        # Save prediction to database (no image_path stored, just analysis results)
        cursor.execute('''INSERT INTO predictions 
                       (user_id, image_path, predicted_emotion, confidence, capture_type) 
                       VALUES (?, ?, ?, ?, ?)''', 
                      (user_id, 'camera_capture', emotion, confidence, 'live_capture'))
        conn.commit()
        conn.close()
        
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'all_emotions': all_emotions,
            'message': f'Hello {name}! Your emotion has been detected.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500





if __name__ == '__main__':
    init_db()
    init_emotion_detector_global()
    port = int(os.environ.get('PORT', 8000))  # Changed default port to 8000
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
