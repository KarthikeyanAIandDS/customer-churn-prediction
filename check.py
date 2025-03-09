import os

if os.path.exists("churn_model.pkl"):
    print("Model file exists.")
else:
    print("Model file NOT found. Check if it was saved correctly.")
