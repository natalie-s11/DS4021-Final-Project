# Split the original dataset
# imports 
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Create Data folder if it doesn't exist
os.makedirs("Data", exist_ok=True)

# Load the dataset
df = pd.read_csv("Credit Risk Benchmark Dataset.csv")

# Get age statistics for pre-requisite assignment 
print("Mean Age:", df["age"].mean())
print("Age Standard Deviation:", df["age"].std())

# Separate features (X) and target (y)
X = df.drop(columns=["dlq_2yrs"])
y = df["dlq_2yrs"]

# Stratified split 80/20 split
X_train, X_report, y_train, y_report = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Recombine features and target
split1_df = pd.concat([X_train, y_train], axis=1)
split2_df = pd.concat([X_report, y_report], axis=1)

# Save CSV files inside Data folder
split1_df.to_csv("Data/training.csv", index=False)
split2_df.to_csv("Data/reporting.csv", index=False)

# Print positive class counts
print("Training positive class count:", (y_train == 1).sum())
print("Reporting positive class count:", (y_report == 1).sum())