from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("churn_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert data into a DataFrame
        feature_names = ['Call Failure', 'Complaints', 'Subscription Length', 
                         'Charge Amount', 'Seconds of Use', 'Frequency of use', 
                         'Frequency of SMS', 'Distinct Called Numbers', 
                         'Age Group', 'Tariff Plan', 'Status', 'Age', 'Customer Value']

        input_data = pd.DataFrame([data], columns=feature_names)

        # Make prediction
        prediction = model.predict(input_data)
        result = "Yes" if prediction[0] == 1 else "No"

        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
