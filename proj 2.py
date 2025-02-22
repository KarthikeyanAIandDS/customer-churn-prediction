import pandas as pd

# Load the dataset
csv_path = "C:/Users/vimal/Downloads/data/data/customer_churn.csv"
churn_df = pd.read_csv(csv_path)

# Display first few rows
print(churn_df.head())
