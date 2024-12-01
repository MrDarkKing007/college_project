I'll create a Flask application in Python that handles these machine learning model files for various cybersecurity tasks. I'll provide a comprehensive setup and implementation with clear comments.



```python
import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize Flask application
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = '/tmp/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'csv'}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained models and vectorizers
def load_models():
    """
    Load all pre-trained machine learning models and vectorizers.
    Handles potential loading errors gracefully.
    """
    models = {}
    model_files = [
        'Malware_classifier.pkl',
        'Malware_features.pkl', 
        'DDos_Model.pkl', 
        'spam_model.pkl', 
        'spam_vectorizer.pkl', 
        'phishing_url.pkl'
    ]
    
    for model_file in model_files:
        try:
            with open(model_file, 'rb') as f:
                models[model_file] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: Model file {model_file} not found.")
        except Exception as e:
            print(f"Error loading {model_file}: {e}")
    
    return models

# Load models when application starts
LOADED_MODELS = load_models()

def allowed_file(filename):
    """
    Check if the uploaded file has an allowed extension.
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/predict/malware', methods=['POST'])
def predict_malware():
    """
    Endpoint for malware prediction.
    Expects file upload or JSON input with features.
    """
    try:
        # Check if file is uploaded
        if 'file' not in request.files:
            # If no file, check for JSON input
            if request.is_json:
                features = request.get_json()
                # Assuming features is a list or numpy array
                malware_model = LOADED_MODELS.get('Malware_classifier.pkl')
                if malware_model:
                    prediction = malware_model.predict(features)
                    return jsonify({
                        'is_malware': bool(prediction[0]),
                        'confidence': float(np.max(malware_model.predict_proba(features)))
                    })
                else:
                    return jsonify({'error': 'Malware model not loaded'}), 500
            
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        # If filename is empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # If file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process file with malware model
            malware_model = LOADED_MODELS.get('Malware_classifier.pkl')
            if malware_model:
                # Add your file processing logic here
                # This is a placeholder - you'll need to implement actual feature extraction
                # prediction = malware_model.predict([extracted_features])
                return jsonify({
                    'message': f'File {filename} uploaded and processed',
                    'is_malware': False  # Placeholder
                })
            else:
                return jsonify({'error': 'Malware model not loaded'}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/ddos', methods=['POST'])
def predict_ddos():
    """
    Endpoint for DDoS attack prediction.
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON input required'}), 400
        
        data = request.get_json()
        ddos_model = LOADED_MODELS.get('DDos_Model.pkl')
        
        if ddos_model:
            prediction = ddos_model.predict([list(data.values())])
            return jsonify({
                'is_ddos': bool(prediction[0]),
                'confidence': float(np.max(ddos_model.predict_proba([list(data.values())])))
            })
        else:
            return jsonify({'error': 'DDoS model not loaded'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/spam', methods=['POST'])
def predict_spam():
    """
    Endpoint for spam detection.
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON input required'}), 400
        
        data = request.get_json()
        spam_model = LOADED_MODELS.get('spam_model.pkl')
        spam_vectorizer = LOADED_MODELS.get('spam_vectorizer.pkl')
        
        if spam_model and spam_vectorizer:
            # Vectorize input text
            vectorized_text = spam_vectorizer.transform([data.get('text', '')])
            prediction = spam_model.predict(vectorized_text)
            
            return jsonify({
                'is_spam': bool(prediction[0]),
                'confidence': float(np.max(spam_model.predict_proba(vectorized_text)))
            })
        else:
            return jsonify({'error': 'Spam models not loaded'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/phishing', methods=['POST'])
def predict_phishing():
    """
    Endpoint for phishing URL detection.
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON input required'}), 400
        
        data = request.get_json()
        phishing_model = LOADED_MODELS.get('phishing_url.pkl')
        
        if phishing_model:
            prediction = phishing_model.predict([data.get('url', '')])
            return jsonify({
                'is_phishing': bool(prediction[0])
            })
        else:
            return jsonify({'error': 'Phishing model not loaded'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    """
    Simple home route to confirm application is running.
    """
    return jsonify({
        'message': 'Cybersecurity ML Prediction Service',
        'available_endpoints': [
            '/predict/malware',
            '/predict/ddos',
            '/predict/spam',
            '/predict/phishing'
        ]
    })

if __name__ == '__main__':
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)

```

Now, I'll provide you with the complete setup instructions for Ubuntu:

1. Update and Prepare Ubuntu:
```bash
sudo apt update
sudo apt upgrade -y
```

2. Install Python and Dependencies:
```bash
sudo apt install python3 python3-pip python3-venv -y
```

3. Create Project Directory and Virtual Environment:
```bash
mkdir cybersecurity-ml-app
cd cybersecurity-ml-app
python3 -m venv venv
source venv/bin/activate
```

4. Install Required Python Packages:
```bash
pip install flask numpy pandas scikit-learn werkzeug
```

5. Save the Python script:
```bash
nano app.py
# Paste the entire code from the artifact above
# Save (Ctrl+O, Enter) and exit (Ctrl+X)
```

6. Ensure Model Files:
- Place all your .pkl files (`Malware_classifier.pkl`, `DDos_Model.pkl`, etc.) in the same directory as `app.py`

7. Run the Application:
```bash
python3 app.py
```

Notes and Recommendations:
- This is a basic implementation. You'll need to replace placeholder prediction logic with your actual model's feature extraction and prediction methods.
- Ensure your model files are compatible and correctly pickled.
- For production, use Gunicorn or uWSGI instead of the Flask development server.
- Add proper error handling and logging for production use.

Endpoints:
- `/predict/malware`: Malware detection
- `/predict/ddos`: DDoS attack detection
- `/predict/spam`: Spam detection
- `/predict/phishing`: Phishing URL detection

Example curl commands for testing:
```bash
# Malware file upload
curl -F "file=@/path/to/suspicious/file" http://localhost:5000/predict/malware

# DDoS prediction
curl -X POST http://localhost:5000/predict/ddos \
     -H "Content-Type: application/json" \
     -d '{"feature1": 10, "feature2": 20}'

# Spam prediction
curl -X POST http://localhost:5000/predict/spam \
     -H "Content-Type: application/json" \
     -d '{"text": "suspicious spam message"}'

# Phishing prediction
curl -X POST http://localhost:5000/predict/phishing \
     -H "Content-Type: application/json" \
     -d '{"url": "http://suspicious-site.com"}'
```

Would you like me to elaborate on any part of the implementation or explain any specific section in more detail?
