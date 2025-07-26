#!/usr/bin/env python3
"""
XYZ Bank Churn Prediction and Retention System
Main Application Script

Author: AI/ML Engineering Team
Date: July 2025
Description: Comprehensive churn prediction and retention system with NLP analysis,
            personalized retention strategies, and customer service chatbot.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from textblob import TextBlob
import pickle
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionSystem:
    """Main class for churn prediction and retention system"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False

    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the customer data"""
        print("Loading and preprocessing data...")

        # Load data
        df = pd.read_excel(file_path)

        # Remove empty columns and handle missing values
        df = df.drop('Recommendation', axis=1, errors='ignore')
        df = df.dropna(subset=['Churn'])

        # Handle missing values
        categorical_cols = ['Dependents', 'Priority Account', 'Credit Cards', 'Loan Account', 
                           'Netbanking', 'TechSupport Availed', 'Zero Balance Account', 
                           'FDs', 'Paperless Banking', 'Category']

        for col in categorical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                df[col].fillna(mode_value, inplace=True)

        # Handle numerical missing values
        numerical_cols = ['tenure in months', 'Monthly Average Balance (USD)']
        for col in numerical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)

        # Convert yearly balance to numeric
        df['Yearly Average Balance (USD)'] = pd.to_numeric(df['Yearly Average Balance (USD)'], errors='coerce')
        df['Yearly Average Balance (USD)'].fillna(df['Yearly Average Balance (USD)'].median(), inplace=True)

        # Feature engineering
        df['Tenure_Group'] = pd.cut(df['tenure in months'], 
                                   bins=[0, 12, 24, 48, 72], 
                                   labels=['New', 'Short-term', 'Medium-term', 'Long-term'])

        df['Balance_Ratio'] = df['Monthly Average Balance (USD)'] / (df['Yearly Average Balance (USD)'] / 12 + 1)

        # Count total services
        service_cols = ['Credit Cards', 'Netbanking', 'Debit Card', 'MobileApp', 'TechSupport Availed', 'FDs']
        df['Total_Services'] = 0
        for col in service_cols:
            df['Total_Services'] += (df[col] == 'Yes').astype(int)

        # Add sentiment analysis
        df['Sentiment_Score'] = df['Customer Feedback'].apply(self._get_sentiment_score)

        return df

    def _get_sentiment_score(self, text):
        """Calculate sentiment score for text"""
        if pd.isna(text) or text == "":
            return 0
        blob = TextBlob(str(text))
        return blob.sentiment.polarity

    def train_model(self, df):
        """Train the churn prediction model"""
        print("Training churn prediction model...")

        # Define feature columns
        self.feature_columns = [
            'Gender', 'Senior Citizen', 'Marital Status', 'Dependents', 'tenure in months',
            'Priority Account', 'Credit Cards', 'Loan Account', 'Netbanking', 'Debit Card',
            'MobileApp', 'TechSupport Availed', 'Zero Balance Account', 'FDs', 'Interest Deposited',
            'Paperless Banking', 'Monthly Average Balance (USD)', 'Yearly Average Balance (USD)',
            'Total_Services', 'Balance_Ratio', 'Sentiment_Score'
        ]

        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

        # Encode categorical variables
        categorical_features = X.select_dtypes(include=['object']).columns
        for col in categorical_features:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        # Handle any remaining NaN values
        X = X.fillna(X.median())

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train Logistic Regression model
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        test_pred = self.model.predict(X_test_scaled)
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]

        accuracy = (test_pred == y_test).mean()
        auc_score = roc_auc_score(y_test, test_proba)

        print(f"Model trained successfully!")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC-ROC: {auc_score:.4f}")

        self.is_trained = True
        return accuracy, auc_score

    def predict_churn_risk(self, customer_data):
        """Predict churn risk for a single customer"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Preprocess customer data
        customer_df = pd.DataFrame([customer_data])

        # Encode categorical variables
        for col in self.label_encoders:
            if col in customer_df.columns:
                try:
                    customer_df[col] = self.label_encoders[col].transform(customer_df[col].astype(str))
                except:
                    customer_df[col] = 0

        # Scale features
        customer_scaled = self.scaler.transform(customer_df[self.feature_columns])

        # Predict
        churn_probability = self.model.predict_proba(customer_scaled)[0][1]
        risk_level = "High" if churn_probability > 0.7 else "Medium" if churn_probability > 0.4 else "Low"

        return {
            'churn_probability': churn_probability,
            'risk_level': risk_level
        }

    def save_model(self, filepath):
        """Save trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load trained model and preprocessors"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = True

        print(f"Model loaded from {filepath}")

class RetentionStrategyGenerator:
    """Generate personalized retention strategies"""

    def generate_strategy(self, customer_profile, prediction_result):
        """Generate personalized retention strategy"""
        strategies = []

        # High-risk customers get immediate attention
        if prediction_result['risk_level'] == "High":
            strategies.append("🚨 IMMEDIATE ACTION REQUIRED")
            strategies.append("• Schedule urgent call with relationship manager")
            strategies.append("• Offer premium service upgrade with 6-month free trial")

        # Sentiment-based strategies
        if customer_profile.get('Sentiment_Score', 0) < -0.1:
            strategies.append("• Address specific complaint mentioned in feedback")
            strategies.append("• Provide personalized apology and compensation")
            strategies.append("• Follow up within 48 hours to ensure satisfaction")

        # Service-based strategies
        if customer_profile.get('Total_Services', 0) < 3:
            strategies.append("• Cross-sell relevant banking products")
            strategies.append("• Provide educational content about unused services")

        # Tenure-based strategies
        if customer_profile.get('tenure in months', 0) < 12:
            strategies.append("• Welcome package for new customers")
            strategies.append("• Assign dedicated onboarding specialist")

        # Default strategy
        if not strategies:
            strategies.append("• Regular check-in call")
            strategies.append("• Customer satisfaction survey")

        return {
            'risk_level': prediction_result['risk_level'],
            'churn_probability': f"{prediction_result['churn_probability']:.1%}",
            'strategies': strategies
        }

class CustomerServiceChatbot:
    """Intelligent customer service chatbot"""

    def __init__(self):
        self.category_routing = {
            'credit card': 'Credit Cards Team',
            'debit card': 'Debit Cards Team', 
            'loan': 'Loans Team',
            'savings account': 'Savings Account Team',
            'current account': 'Current Account Team',
            'atm': 'ATM Services Team',
            'mobile banking': 'Mobile Banking Team',
            'online banking': 'Online Banking Team',
            'branch service': 'Branch Services Team',
            'fixed deposit': 'Fixed Deposit Team'
        }

    def classify_and_respond(self, query):
        """Classify query and generate response"""
        query_lower = query.lower()

        # Determine category
        detected_category = None
        for category, team in self.category_routing.items():
            if category in query_lower:
                detected_category = category
                break

        # Generate response
        response = f"Thank you for contacting us regarding your {detected_category or 'banking'} query. "

        if 'blocked' in query_lower:
            response += "I understand you're experiencing card blocking issues. "
        elif 'not working' in query_lower:
            response += "I understand you're experiencing service issues. "
        elif 'charges' in query_lower or 'fees' in query_lower:
            response += "I understand you have questions about charges. "

        routing_team = self.category_routing.get(detected_category, 'General Support Team')
        response += f"I'm routing your request to our {routing_team} who will assist you further. "

        if detected_category in ['credit card', 'debit card']:
            response += "You can expect a response within 30 minutes for urgent card issues."
        else:
            response += "You can expect a response within 2 hours."

        return {
            'response': response,
            'routing_team': routing_team,
            'category': detected_category
        }

def main():
    """Main function to demonstrate the system"""
    print("XYZ Bank Churn Prediction and Retention System")
    print("=" * 60)

    # Initialize system
    churn_system = ChurnPredictionSystem()
    retention_generator = RetentionStrategyGenerator()
    chatbot = CustomerServiceChatbot()

    # Load and train model (in production, model would be pre-trained)
    try:
        df = churn_system.load_and_preprocess_data('customer-churn-data_usecase2_Hackathon.xlsx')
        accuracy, auc = churn_system.train_model(df)

        # Demo prediction
        sample_customer = {
            'Gender': 'Female',
            'Senior Citizen': 0,
            'Marital Status': 'No',
            'Dependents': 'Yes',
            'tenure in months': 5,
            'Priority Account': 'Yes',
            'Credit Cards': 'No',
            'Loan Account': 'general loan',
            'Netbanking': 'No',
            'Debit Card': 'Yes',
            'MobileApp': 'No',
            'TechSupport Availed': 'No',
            'Zero Balance Account': 'No',
            'FDs': 'No',
            'Interest Deposited': 'Month-to-month',
            'Paperless Banking': 'Yes',
            'Monthly Average Balance (USD)': 45.0,
            'Yearly Average Balance (USD)': 540.0,
            'Total_Services': 2,
            'Balance_Ratio': 1.0,
            'Sentiment_Score': -0.3
        }

        # Predict churn risk
        prediction = churn_system.predict_churn_risk(sample_customer)
        print(f"\nChurn Prediction Demo:")
        print(f"Risk Level: {prediction['risk_level']}")
        print(f"Churn Probability: {prediction['churn_probability']:.1%}")

        # Generate retention strategy
        strategy = retention_generator.generate_strategy(sample_customer, prediction)
        print(f"\nRetention Strategy:")
        for item in strategy['strategies']:
            print(item)

        # Chatbot demo
        print(f"\nChatbot Demo:")
        query = "My Debit Card often gets blocked without reason"
        response = chatbot.classify_and_respond(query)
        print(f"Customer: {query}")
        print(f"Chatbot: {response['response']}")

        print(f"\n✅ System demonstration completed successfully!")

    except FileNotFoundError:
        print("Data file not found. Please ensure 'customer-churn-data_usecase2_Hackathon.xlsx' is in the current directory.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
