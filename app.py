from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os  # Import os to get PORT for Render

app = Flask(__name__)

# Load the trained model
model = joblib.load("churn_model.pkl")

# Define feature names expected by the model
FEATURE_NAMES = [
    "Call Failure", "Complaints", "Subscription Length", 
    "Charge Amount", "Seconds of Use", "Frequency of use", 
    "Frequency of SMS", "Distinct Called Numbers", 
    "Age Group", "Tariff Plan", "Status", "Age", "Customer Value"
]

# Home Route (for Browser)
@app.route('/')
def home():
    return "Welcome to the Customer Churn Prediction API! Use /predict to make predictions."

# Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Debugging: Print received data
        print("Received Data:", data)

        # Validate input
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format. Expected JSON object."}), 400

        # Ensure all required features are present
        missing_features = [feature for feature in FEATURE_NAMES if feature not in data]
        if missing_features:
            return jsonify({"error": f"Missing features: {missing_features}"}), 400

        # Convert data to a DataFrame
        input_data = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Make prediction
        prediction = model.predict(input_data)
        result = "Yes" if prediction[0] == 1 else "No"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Error Handling
@app.errorhandler(400)
def bad_request(error):
    return jsonify({"error": "Bad Request - Check your JSON input"}), 400

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal Server Error - Please try again later"}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Use Render's port
    app.run(host="0.0.0.0", port=port)
