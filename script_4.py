# Fix NLTK download and implement simpler NLP analysis
import nltk

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('stopwords')

# Now implement a simpler NLP approach
print("="*70)
print("NLP ANALYSIS FOR CUSTOMER FEEDBACK (SIMPLIFIED APPROACH)")
print("="*70)

# Simplified sentiment analysis using TextBlob
def simple_sentiment_analysis(text):
    if pd.isna(text) or text == "":
        return 0, 'neutral'
    
    blob = TextBlob(str(text))
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return polarity, 'positive'
    elif polarity < -0.1:
        return polarity, 'negative'
    else:
        return polarity, 'neutral'

print("1. Customer Feedback Analysis...")
print(f"Total feedback entries: {len(df_processed['Customer Feedback'])}")
print(f"Sample feedback entries:")
for i in range(3):
    print(f"- {df_processed['Customer Feedback'].iloc[i]}")

# Apply sentiment analysis
sentiment_results = df_processed['Customer Feedback'].apply(simple_sentiment_analysis)
df_processed['Sentiment_Score'] = [result[0] for result in sentiment_results]
df_processed['Sentiment_Label'] = [result[1] for result in sentiment_results]

print("\n2. Sentiment Distribution:")
sentiment_dist = df_processed['Sentiment_Label'].value_counts()
for sentiment, count in sentiment_dist.items():
    percentage = (count / len(df_processed)) * 100
    print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")

print("\n3. Sentiment vs Churn Analysis:")
sentiment_churn = pd.crosstab(df_processed['Sentiment_Label'], df_processed['Churn'], normalize='index') * 100
print("Churn Rate by Sentiment (%):")
print(sentiment_churn.round(2))

print("\n4. Category-wise Analysis:")
category_sentiment = pd.crosstab(df_processed['Category'], df_processed['Sentiment_Label'], normalize='index') * 100
print("Top 5 Categories by Sentiment Distribution (%):")
print(category_sentiment.round(1).head())

print("\n5. Key Issues Analysis:")
# Simple keyword analysis for negative feedback
negative_feedback = df_processed[df_processed['Sentiment_Label'] == 'negative']['Customer Feedback']
print(f"Total negative feedback entries: {len(negative_feedback)}")

# Common complaint patterns
complaint_patterns = {
    'fees': ['fee', 'charge', 'cost', 'expensive', 'high'],
    'service_issues': ['slow', 'delay', 'wait', 'long', 'crowded'],
    'technical_problems': ['not work', 'block', 'decline', 'error', 'crash'],
    'process_issues': ['complicated', 'difficult', 'confusing', 'cumbersome'],
    'accessibility': ['accessible', 'disabled', 'seating', 'staff'],
    'reliability': ['unreliable', 'down', 'out of service', 'frequent']
}

print("\nKey complaint categories:")
for category, keywords in complaint_patterns.items():
    count = 0
    for feedback in negative_feedback:
        if any(keyword in str(feedback).lower() for keyword in keywords):
            count += 1
    if count > 0:
        percentage = (count / len(negative_feedback)) * 100
        print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

# Most common specific issues
print("\n6. Most Common Specific Issues (Top 10):")
issue_counts = {}
for feedback in df_processed['Customer Feedback']:
    feedback_str = str(feedback).lower()
    if 'too high' in feedback_str:
        issue_counts['High fees/charges'] = issue_counts.get('High fees/charges', 0) + 1
    if 'not work' in feedback_str:
        issue_counts['Not working'] = issue_counts.get('Not working', 0) + 1
    if 'slow' in feedback_str:
        issue_counts['Slow service'] = issue_counts.get('Slow service', 0) + 1
    if 'crowd' in feedback_str:
        issue_counts['Crowded branches'] = issue_counts.get('Crowded branches', 0) + 1
    if 'complicated' in feedback_str:
        issue_counts['Complicated processes'] = issue_counts.get('Complicated processes', 0) + 1
    if 'confusing' in feedback_str:
        issue_counts['Confusing procedures'] = issue_counts.get('Confusing procedures', 0) + 1
    if 'delay' in feedback_str:
        issue_counts['Delayed services'] = issue_counts.get('Delayed services', 0) + 1
    if 'unreliable' in feedback_str:
        issue_counts['Unreliable services'] = issue_counts.get('Unreliable services', 0) + 1

sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for issue, count in sorted_issues:
    print(f"  {issue}: {count} mentions")