import pandas as pd

# Load the dataset
csv_path = "C:/Users/vimal/Downloads/data/data/customer_churn.csv"
churn_df = pd.read_csv(csv_path)

# Check if the dataset is loaded correctly
print(churn_df.head())  # Display first few rows
print(churn_df.columns)  # Print column names

from imblearn.over_sampling import SMOTE

X = churn_df.drop(columns=['Churn'])  # Features
y = churn_df['Churn']                 # Target variable

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Print new class distribution
print(y_resampled.value_counts())
from sklearn.model_selection import train_test_split

# Features & target
X = churn_df.drop(columns=['Churn'])  # Independent variables
y = churn_df['Churn']                 # Target variable (0 or 1)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check the distribution
print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import numpy as np

feature_importances = model.feature_importances_
feature_names = X_train.columns

indices = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10,5))
plt.title("Feature Importance")
plt.bar(range(X_train.shape[1]), feature_importances[indices], align="center")
plt.xticks(range(X_train.shape[1]), np.array(feature_names)[indices], rotation=90)
plt.show()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Define the model
rf = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Perform Grid Search
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Evaluate on test data
best_rf = grid_search.best_estimator_
accuracy = best_rf.score(X_test, y_test)
print("Optimized Model Accuracy:", accuracy)

import joblib

joblib.dump(model, "churn_model.pkl")
print("Model saved successfully!")




