# Data Preprocessing and Cleaning
print("="*60)
print("DATA PREPROCESSING AND CLEANING")
print("="*60)

# Create a copy for preprocessing
df_processed = df.copy()

# 1. Handle missing values
print("1. Handling Missing Values...")

# Remove the 'Recommendation' column as it's completely empty
df_processed = df_processed.drop('Recommendation', axis=1)

# Handle missing values in target variable (Churn) - remove rows with missing churn
df_processed = df_processed.dropna(subset=['Churn'])
print(f"Removed {len(df) - len(df_processed)} rows with missing Churn values")

# Handle missing values in other columns
# For categorical variables, fill with mode
categorical_cols = ['Dependents', 'Priority Account', 'Credit Cards', 'Loan Account', 
                   'Netbanking', 'TechSupport Availed', 'Zero Balance Account', 
                   'FDs', 'Paperless Banking', 'Category']

for col in categorical_cols:
    if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
        mode_value = df_processed[col].mode().iloc[0] if len(df_processed[col].mode()) > 0 else 'Unknown'
        df_processed[col].fillna(mode_value, inplace=True)
        print(f"Filled {col} missing values with mode: {mode_value}")

# For numerical variables, fill with median
numerical_cols = ['tenure in months', 'Monthly Average Balance (USD)']
for col in numerical_cols:
    if col in df_processed.columns and df_processed[col].isnull().sum() > 0:
        median_value = df_processed[col].median()
        df_processed[col].fillna(median_value, inplace=True)
        print(f"Filled {col} missing values with median: {median_value}")

# Handle 'Yearly Average Balance (USD)' - convert to numeric after cleaning
df_processed['Yearly Average Balance (USD)'] = pd.to_numeric(df_processed['Yearly Average Balance (USD)'], errors='coerce')
yearly_balance_median = df_processed['Yearly Average Balance (USD)'].median()
df_processed['Yearly Average Balance (USD)'].fillna(yearly_balance_median, inplace=True)
print(f"Converted and filled Yearly Average Balance missing values with median: {yearly_balance_median}")

print(f"\nDataset shape after preprocessing: {df_processed.shape}")
print(f"Missing values after preprocessing: {df_processed.isnull().sum().sum()}")

# 2. Feature Engineering
print("\n2. Feature Engineering...")

# Create new features
df_processed['Tenure_Group'] = pd.cut(df_processed['tenure in months'], 
                                     bins=[0, 12, 24, 48, 72], 
                                     labels=['New', 'Short-term', 'Medium-term', 'Long-term'])

# Balance ratio
df_processed['Balance_Ratio'] = df_processed['Monthly Average Balance (USD)'] / (df_processed['Yearly Average Balance (USD)'] / 12 + 1)

# Total services count
service_cols = ['Credit Cards', 'Netbanking', 'Debit Card', 'MobileApp', 'TechSupport Availed', 'FDs']
df_processed['Total_Services'] = 0
for col in service_cols:
    df_processed['Total_Services'] += (df_processed[col] == 'Yes').astype(int)

print("Created new features: Tenure_Group, Balance_Ratio, Total_Services")

# Display final preprocessed data info
print("\n3. Final Preprocessed Dataset Info:")
print(f"Shape: {df_processed.shape}")
print(f"Churn distribution: {df_processed['Churn'].value_counts().to_dict()}")