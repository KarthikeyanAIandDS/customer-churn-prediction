import joblib

# Assuming `model` is your trained classifier
joblib.dump(model, "churn_model.pkl")
print("Model saved successfully!")
