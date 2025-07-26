# Configuration file for XYZ Bank Churn Prediction System

# Model Configuration
MODEL_CONFIG = {
    "algorithm": "LogisticRegression",
    "random_state": 42,
    "max_iter": 1000,
    "test_size": 0.2
}

# Risk Thresholds
RISK_THRESHOLDS = {
    "high_risk": 0.7,
    "medium_risk": 0.4
}

# Feature Configuration
FEATURE_COLUMNS = [
    'Gender', 'Senior Citizen', 'Marital Status', 'Dependents', 'tenure in months',
    'Priority Account', 'Credit Cards', 'Loan Account', 'Netbanking', 'Debit Card',
    'MobileApp', 'TechSupport Availed', 'Zero Balance Account', 'FDs', 'Interest Deposited',
    'Paperless Banking', 'Monthly Average Balance (USD)', 'Yearly Average Balance (USD)',
    'Total_Services', 'Balance_Ratio', 'Sentiment_Score'
]

# Chatbot Configuration
CHATBOT_CONFIG = {
    "response_time_card_issues": "30 minutes",
    "response_time_general": "2 hours",
    "categories": {
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
}

# Database Configuration (for production)
DATABASE_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "churn_prediction",
    "username": "admin",
    "password": "password"  # Use environment variable in production
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "debug": False
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "churn_system.log"
}
