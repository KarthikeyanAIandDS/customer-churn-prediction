from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("churn_model.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        data = request.get_json(force=True)
        features = np.array([data['features']])  # Convert to NumPy array
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]  # Get churn probability
        
        # Confidence Score
        confidence = round(probability * 100, 2)

        # Response message
        result = {
            "Churn": "Yes" if prediction == 1 else "No",
            "Probability": f"{confidence}%",
            "Message": "High chance of churn, consider retention strategies!" if prediction == 1 else "Customer is likely to stay."
        }

        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
