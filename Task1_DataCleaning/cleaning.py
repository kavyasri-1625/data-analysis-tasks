import pandas as pd

# Load dataset
df = pd.read_csv("customers-100.csv")

print("Original Shape:", df.shape)

# Remove duplicates
df = df.drop_duplicates()

# Handle missing values
df = df.dropna(subset=["Customer Id"])

# Fill numeric missing values
for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

# Convert date columns if present
for col in df.columns:
    if "date" in col.lower():
        df[col] = pd.to_datetime(df[col], errors="coerce")

# Clean text columns
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.lower().str.strip()

# Save cleaned file
df.to_csv("cleaned_data.csv", index=False)

print("Cleaned Shape:", df.shape)
print("Cleaning Completed ✅")