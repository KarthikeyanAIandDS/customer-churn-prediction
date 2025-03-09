import joblib
import numpy as np
import pandas as pd

# Load the model
try:
    model = joblib.load("churn_model.pkl")
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)

# Define feature names (ensure they match training data)
feature_names = [
    'Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount',
    'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
    'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age',
    'Customer Value'
]

# Sample input (ensure order matches dataset)
sample_input = np.array([[5, 0, 35, 75.5, 300, 50, 10, 20, 1, 2, 0, 25, 200.5]])
sample_df = pd.DataFrame(sample_input, columns=feature_names)

# Make prediction
try:
    prediction = model.predict(sample_df)
    print("Predicted Churn:", "Yes" if prediction[0] == 1 else "No")
except Exception as e:
    print("Prediction failed:", e)
