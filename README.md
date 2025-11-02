# Emotion Detection Web App 🎭

A simple, lightweight emotion detection web application using machine learning.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python app.py
   ```

3. **Open your browser:**
   ```
   http://localhost:8000
   ```

## 📁 Project Structure

```
EMOTION_DETECTOR/
├── app.py                      # Main Flask application
├── model.py                    # Emotion detector wrapper
├── simple_emotion_detector.py  # ML model implementation
├── requirements.txt            # Python dependencies
├── Procfile                   # Deployment configuration
├── templates/
│   └── index.html            # Single-page web interface
├── static/
│   └── style.css            # CSS styling
├── models/
│   └── simple_emotion_model.joblib  # Trained ML model
├── emotion_detection.db      # SQLite database
└── uploads/                  # File upload directory
```

## ✨ Features

- **Real Emotion Detection**: Uses scikit-learn with OpenCV face detection
- **Single-Page Interface**: Upload images and see results instantly
- **7 Emotions**: angry, disgust, fear, happy, sad, surprise, neutral
- **Database Storage**: SQLite for user data and predictions
- **Responsive Design**: Works on desktop and mobile devices

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **ML**: scikit-learn, OpenCV, numpy
- **Frontend**: HTML5, Bootstrap 5, JavaScript
- **Database**: SQLite
- **Deployment**: Heroku-ready with Procfile

## 🎯 How It Works

1. **Upload an image** with a clear face
2. **Face detection** using OpenCV algorithms
3. **Feature extraction** from the detected face region
4. **Emotion prediction** using trained Random Forest model
5. **Results display** with confidence score

## 🔧 Development

### Local Setup
```bash
# Clone and navigate to directory
cd /path/to/AKINBOYEWA_23CG034029

# Install dependencies
pip3 install -r requirements.txt

# Run application
python3 app.py
```

### Deploy to Heroku
```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main
```

## 📊 Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: Histogram + Local Binary Pattern-like features
- **Input**: 48x48 grayscale face images
- **Accuracy**: ~18% (on synthetic demo data)
- **Model Size**: ~1MB (joblib format)

## 📝 API Endpoints

- `GET /` - Main web interface
- `POST /upload` - Process uploaded image
- `POST /live_capture` - Process camera capture

## 🔐 Security

- File upload validation (16MB limit)
- SQL injection protection
- Input sanitization
- Temporary file cleanup

## 📱 Usage

1. **Image Upload**: Click "Choose File" and select a photo
2. **Analysis**: Click "Analyze Emotion" to detect emotions
3. **Results**: View detected emotion and confidence score
4. **History**: All predictions are saved in the database

## 🎨 Customization

Edit `templates/index.html` for UI changes or `static/style.css` for styling.

## 📄 License

Educational/Demo project - free to use and modify.

---

**Simple. Fast. Effective.** 🚀