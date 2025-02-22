import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("churn_model.pkl")

# Define feature names (ensure they match the dataset used for training)
feature_names = [
    'Call Failure', 'Complaints', 'Subscription Length', 'Charge Amount',
    'Seconds of Use', 'Frequency of use', 'Frequency of SMS',
    'Distinct Called Numbers', 'Age Group', 'Tariff Plan', 'Status', 'Age',
    'Customer Value'
]

# Sample test data (ensure the order matches your dataset)
sample_input = np.array([[5, 0, 35, 75.5, 300, 50, 10, 20, 1, 2, 0, 25, 200.5]])

# Convert to DataFrame to include feature names
sample_df = pd.DataFrame(sample_input, columns=feature_names)

# Make a prediction
prediction = model.predict(sample_df)

# Display result
print("Predicted Churn:", "Yes" if prediction[0] == 1 else "No")
