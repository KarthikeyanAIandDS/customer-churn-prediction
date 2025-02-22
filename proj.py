import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
csv_path = r"C:\Users\vimal\Downloads\data\data\customer_churn.csv"  # Ensure the correct path
churn_df = pd.read_csv(csv_path)

# Check for missing values
print(churn_df.isnull().sum())

# Check class balance
print(churn_df['Churn'].value_counts())

# Correlation heatmap
plt.figure(figsize=(12,6))
sns.heatmap(churn_df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

