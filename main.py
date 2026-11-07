import pandas as pd

df = pd.read_csv("datasets/synthetic_medical_symptoms_dataset.csv")

# Count records for each disease
print(df["diagnosis"].value_counts())

# Calculate percentage distribution
print(df["diagnosis"].value_counts(normalize=True) * 100)