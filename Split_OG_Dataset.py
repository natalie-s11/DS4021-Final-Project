# Split the original dataset
# imports 
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Create Data folder if it doesn't exist
os.makedirs("Data", exist_ok=True)

# Load the dataset
df = pd.read_csv("Credit Risk Benchmark Dataset.csv")

# Separate features (X) and target (y)
X = df.drop(columns=["dlq_2yrs"])
y = df["dlq_2yrs"]

# Stratified split 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Recombine features and target
split1_df = pd.concat([X_train, y_train], axis=1)
split2_df = pd.concat([X_test, y_test], axis=1)

# Save CSV files inside Data folder
split1_df.to_csv("Data/training.csv", index=False)
split2_df.to_csv("Data/testing.csv", index=False)

# Print positive class counts
print("Training positive class count:", (y_train == 1).sum())
print("Testing positive class count:", (y_test == 1).sum())