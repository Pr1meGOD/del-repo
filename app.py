from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(_name_)

# Load the trained model, scaler, and feature names
try:
    with open('house_price_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model = model_data['model']
        scaler = model_data['scaler']
        feature_names = model_data['feature_names']
    print("Model loaded successfully!")
    print(f"Features: {feature_names}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    """Render the home page with the prediction form"""
    return render_template('index.html', features=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please check if house_price_model.pkl exists.'})
        
        # Get form data
        if request.is_json:
            # Handle JSON requests
            data = request.get_json()
        else:
            # Handle form data
            data = request.form.to_dict()
        
        # Extract features from the form data
        features = []
        feature_values = {}
        
        for feature in feature_names:
            value = float(data.get(feature, 0))
            features.append(value)
            feature_values[feature] = value
        
        # Convert to numpy array and reshape
        input_data = np.array(features).reshape(1, -1)
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Round to 2 decimal places
        prediction = round(prediction, 2)
        
        if request.is_json:
            return jsonify({
                'prediction': prediction,
                'input_features': feature_values,
                'status': 'success'
            })
        else:
            return render_template('index.html', 
                                 prediction=prediction, 
                                 features=feature_names,
                                 input_values=feature_values)
    
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        print(error_msg)
        if request.is_json:
            return jsonify({'error': error_msg, 'status': 'error'})
        else:
            return render_template('index.html', 
                                 error=error_msg, 
                                 features=feature_names)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Extract features
        features = []
        for feature in feature_names:
            if feature not in data:
                return jsonify({'error': f'Missing feature: {feature}'}), 400
            features.append(float(data[feature]))
        
        # Make prediction
        input_data = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'prediction': round(prediction, 2),
            'input_features': dict(zip(feature_names, features)),
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/model_info')
def model_info():
    """Get information about the loaded model"""
    if model is None:
        return jsonify({'error': 'Model not loaded'})
    
    return jsonify({
        'features': feature_names,
        'model_type': 'Linear Regression',
        'num_features': len(feature_names),
        'status': 'loaded'
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html', error="Page not found", features=feature_names), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('index.html', error="Internal server error", features=feature_names), 500

if _name_ == '_main_':
    print("Starting Flask application...")
    print("Make sure 'house_price_model.pkl' is in the same directory as this file.")
    print("Navigate to http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, host='0.0.0.0', port=5000)