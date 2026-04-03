import pandas as pd

# Path to your dataset
file_path = r"C:\Users\bhava\OneDrive\Desktop\demo.cpp\Truthlens Ai\data\unified_dataset_test.csv"

# Load dataset
df = pd.read_csv(file_path)

# Extract column names
columns = df.columns.tolist()

# Print them
print("Column Names:")
for col in columns:
    print(col)

# Optional: save column names to a file
with open("dataset_columns.txt", "w") as f:
    for col in columns:
        f.write(col + "\n")