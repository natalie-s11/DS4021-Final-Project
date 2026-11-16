# split the og dataset

import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load your dataset
df = pd.read_csv("Credit Risk Benchmark Dataset.csv")

# 2. Separate features (X) and target (y)
X = df.drop(columns=["dlq_2yrs"])
y = df["dlq_2yrs"]

# 3. Stratified split to ensure even positive class distribution
# test_size=0.5 splits the data in half
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Recombine features and target for each split
split1_df = pd.concat([X_train, y_train], axis=1)
split2_df = pd.concat([X_test, y_test], axis=1)

# 5. Save to separate CSV files
split1_df.to_csv("training.csv", index=False)
split2_df.to_csv("testing.csv", index=False)

print("Split 1 positive class count:", (y_train == 1).sum())
print("Split 2 positive class count:", (y_test == 1).sum())