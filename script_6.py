# Generative AI for Retention Strategies and Chatbot
print("="*70)
print("GENERATIVE AI FOR RETENTION STRATEGIES AND CHATBOT")
print("="*70)

import random

class ChurnRetentionSystem:
    def __init__(self, model_data, df_processed):
        self.model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.df_processed = df_processed
        
    def predict_churn_risk(self, customer_data):
        """Predict churn risk for a customer"""
        # Preprocess customer data
        customer_df = pd.DataFrame([customer_data])
        
        # Encode categorical variables
        for col in self.label_encoders:
            if col in customer_df.columns:
                try:
                    customer_df[col] = self.label_encoders[col].transform(customer_df[col].astype(str))
                except:
                    # Handle unseen categories
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
    
    def generate_retention_strategy(self, customer_profile, prediction_result):
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
            
        # Balance-based strategies
        if customer_profile.get('Monthly Average Balance (USD)', 0) > 80:
            strategies.append("• Offer wealth management consultation")
            strategies.append("• Provide exclusive investment opportunities")
        
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
    def __init__(self):
        # Category routing mapping
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
        
        # Common issues and responses
        self.issue_patterns = {
            'blocked': 'card blocking issues',
            'declined': 'transaction problems',
            'not working': 'service outage',
            'slow': 'performance issues',
            'fees': 'billing inquiries',
            'charges': 'billing inquiries',
            'password': 'account access issues',
            'login': 'account access issues',
            'balance': 'account information',
            'statement': 'account statements'
        }
    
    def classify_query(self, query):
        """Classify customer query and route to appropriate team"""
        query_lower = query.lower()
        
        # Determine category
        detected_category = None
        for category, team in self.category_routing.items():
            if category in query_lower:
                detected_category = category
                break
        
        # Determine issue type
        detected_issue = None
        for keyword, issue_type in self.issue_patterns.items():
            if keyword in query_lower:
                detected_issue = issue_type
                break
        
        return {
            'category': detected_category,
            'issue_type': detected_issue,
            'routing_team': self.category_routing.get(detected_category, 'General Support Team')
        }
    
    def generate_response(self, query):
        """Generate response to customer query"""
        classification = self.classify_query(query)
        
        # Base response
        response = f"Thank you for contacting us regarding your {classification['category'] or 'banking'} query. "
        
        # Add issue-specific response
        if classification['issue_type']:
            response += f"I understand you're experiencing {classification['issue_type']}. "
        
        # Routing information
        response += f"I'm routing your request to our {classification['routing_team']} "
        response += "who will assist you further. "
        
        # Add timeline and follow-up
        if classification['category'] in ['credit card', 'debit card']:
            response += "You can expect a response within 30 minutes for urgent card issues. "
        else:
            response += "You can expect a response within 2 hours. "
        
        response += "Is there anything else I can help clarify about your request?"
        
        return {
            'response': response,
            'routing_team': classification['routing_team'],
            'category': classification['category'],
            'issue_type': classification['issue_type']
        }
    
    def ask_clarifying_question(self, query):
        """Generate clarifying questions when needed"""
        query_lower = query.lower()
        
        if len(query.split()) < 3:  # Very short query
            return "Could you please provide more details about the specific issue you're experiencing?"
        
        if 'not working' in query_lower:
            return "What exactly is not working? Are you getting any error messages?"
        
        if 'blocked' in query_lower:
            return "When did you first notice the blocking? Have you tried any troubleshooting steps?"
        
        if 'charges' in query_lower or 'fees' in query_lower:
            return "Which specific charges are you concerned about? Do you have the transaction date or amount?"
        
        return None

# Initialize the system
retention_system = ChurnRetentionSystem(model_data, df_processed)
chatbot = CustomerServiceChatbot()

print("1. Retention Strategy Generation Demo...")

# Demo with a high-risk customer
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

prediction = retention_system.predict_churn_risk(sample_customer)
strategy = retention_system.generate_retention_strategy(sample_customer, prediction)

print(f"Customer Risk Level: {strategy['risk_level']}")
print(f"Churn Probability: {strategy['churn_probability']}")
print("\nPersonalized Retention Strategy:")
for strategy_item in strategy['strategies']:
    print(strategy_item)

print("\n2. Chatbot Demo...")

# Demo queries
demo_queries = [
    "My Debit Card often gets blocked without reason",
    "The Current Account charges are too high",
    "Login not working",
    "ATM fees",
    "Mobile app crashes"
]

print("Sample Customer Service Interactions:")
for query in demo_queries:
    print(f"\nCustomer: {query}")
    response = chatbot.generate_response(query)
    print(f"Chatbot: {response['response']}")
    
    # Check if clarifying question needed
    clarifying_q = chatbot.ask_clarifying_question(query)
    if clarifying_q:
        print(f"Chatbot (Follow-up): {clarifying_q}")

print(f"\n✅ Retention System and Chatbot ready for deployment!")