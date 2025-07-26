# Detailed exploratory data analysis
print("="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)
missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({
    'Column': missing_values.index,
    'Missing Count': missing_values.values,
    'Missing Percentage': missing_percentage.values
})
missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
print(missing_df)

print("\n" + "="*60)
print("TARGET VARIABLE ANALYSIS")
print("="*60)
# Analyze churn distribution
churn_counts = df['Churn'].value_counts(dropna=False)
print("Churn Distribution:")
print(churn_counts)
print(f"\nChurn Rate: {(churn_counts.get('Yes', 0) / churn_counts.sum()) * 100:.2f}%")

print("\n" + "="*60)
print("CATEGORICAL FEATURES ANALYSIS")
print("="*60)
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('customerID')  # Remove ID column
categorical_cols.remove('Customer Feedback')  # Will analyze separately

for col in categorical_cols[:5]:  # Show first 5 categorical columns
    print(f"\n{col}:")
    print(df[col].value_counts(dropna=False).head())

print("\n" + "="*60)
print("NUMERICAL FEATURES ANALYSIS")
print("="*60)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print("Numerical columns:", numerical_cols)
print("\nDescriptive Statistics:")
print(df[numerical_cols].describe())