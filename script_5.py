# Churn Prediction Models
print("="*70)
print("CHURN PREDICTION MODELS")
print("="*70)

# Prepare data for modeling
print("1. Preparing Data for Modeling...")

# Select features for modeling
feature_columns = [
    'Gender', 'Senior Citizen', 'Marital Status', 'Dependents', 'tenure in months',
    'Priority Account', 'Credit Cards', 'Loan Account', 'Netbanking', 'Debit Card',
    'MobileApp', 'TechSupport Availed', 'Zero Balance Account', 'FDs', 'Interest Deposited',
    'Paperless Banking', 'Monthly Average Balance (USD)', 'Yearly Average Balance (USD)',
    'Total_Services', 'Balance_Ratio', 'Sentiment_Score'
]

# Create features dataset
X = df_processed[feature_columns].copy()
y = df_processed['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

print(f"Features shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# Encode categorical variables
label_encoders = {}
categorical_features = X.select_dtypes(include=['object']).columns

for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"Encoded {len(categorical_features)} categorical features")

# Handle any remaining NaN values
X = X.fillna(X.median())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n2. Model Development and Training...")

# Model 1: Random Forest
print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)

# Predictions
rf_train_pred = rf_model.predict(X_train)
rf_test_pred = rf_model.predict(X_test)
rf_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Model 2: Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)
lr_test_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

print("\n3. Model Evaluation...")

# Evaluation function
def evaluate_model(y_true, y_pred, y_proba, model_name):
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {((y_pred == y_true).sum() / len(y_true)):.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    return roc_auc_score(y_true, y_proba)

# Evaluate models
rf_auc = evaluate_model(y_test, rf_test_pred, rf_test_proba, "Random Forest")
lr_auc = evaluate_model(y_test, lr_test_pred, lr_test_proba, "Logistic Regression")

print(f"\n4. Model Comparison:")
print(f"Random Forest AUC: {rf_auc:.4f}")
print(f"Logistic Regression AUC: {lr_auc:.4f}")

best_model = rf_model if rf_auc > lr_auc else lr_model
best_model_name = "Random Forest" if rf_auc > lr_auc else "Logistic Regression"
print(f"Best Model: {best_model_name}")

print("\n5. Feature Importance Analysis...")
if rf_auc > lr_auc:
    # Random Forest feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
else:
    # Logistic Regression coefficients (absolute values)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': abs(lr_model.coef_[0])
    }).sort_values('importance', ascending=False)

print("Top 10 Most Important Features:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Save models and preprocessors for later use
import pickle

model_data = {
    'best_model': best_model,
    'model_name': best_model_name,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_columns': feature_columns,
    'feature_importance': feature_importance
}

print(f"\nModels trained and ready for deployment!")
print(f"Churn prediction accuracy: {rf_auc:.1%}" if rf_auc > lr_auc else f"Churn prediction accuracy: {lr_auc:.1%}")