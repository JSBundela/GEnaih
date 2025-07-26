# NLP Analysis for Customer Feedback
print("="*70)
print("NLP ANALYSIS FOR CUSTOMER FEEDBACK")
print("="*70)

import re
from collections import Counter
from textblob import TextBlob
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Initialize NLP components
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

print("1. Customer Feedback Overview...")
print(f"Total feedback entries: {len(df_processed['Customer Feedback'])}")
print(f"Sample feedback entries:")
for i in range(3):
    print(f"- {df_processed['Customer Feedback'].iloc[i]}")

# Text preprocessing function
def preprocess_text(text):
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and stem
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words and len(token) > 2]
    return ' '.join(tokens)

# Apply text preprocessing
df_processed['Feedback_Processed'] = df_processed['Customer Feedback'].apply(preprocess_text)

print("\n2. Sentiment Analysis...")
def get_sentiment(text):
    if pd.isna(text) or text == "":
        return 0, 'neutral'
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return polarity, 'positive'
    elif polarity < -0.1:
        return polarity, 'negative'
    else:
        return polarity, 'neutral'

# Apply sentiment analysis
sentiment_results = df_processed['Customer Feedback'].apply(get_sentiment)
df_processed['Sentiment_Score'] = [result[0] for result in sentiment_results]
df_processed['Sentiment_Label'] = [result[1] for result in sentiment_results]

# Sentiment distribution
sentiment_dist = df_processed['Sentiment_Label'].value_counts()
print("Sentiment Distribution:")
for sentiment, count in sentiment_dist.items():
    percentage = (count / len(df_processed)) * 100
    print(f"  {sentiment.capitalize()}: {count} ({percentage:.1f}%)")

print("\n3. Sentiment vs Churn Analysis...")
sentiment_churn = pd.crosstab(df_processed['Sentiment_Label'], df_processed['Churn'], normalize='index') * 100
print("Churn Rate by Sentiment (%):")
print(sentiment_churn.round(2))

print("\n4. Category-wise Sentiment Analysis...")
category_sentiment = pd.crosstab(df_processed['Category'], df_processed['Sentiment_Label'], normalize='index') * 100
print("Sentiment Distribution by Category (%):")
print(category_sentiment.round(1).head())

print("\n5. Key Issues Extraction...")
# Extract common negative feedback themes
negative_feedback = df_processed[df_processed['Sentiment_Label'] == 'negative']['Customer Feedback']
all_negative_text = ' '.join(negative_feedback.astype(str))

# Find common complaint keywords
complaint_keywords = [
    'high', 'slow', 'poor', 'bad', 'difficult', 'complicated', 'issue', 'problem',
    'not working', 'blocked', 'declined', 'fees', 'charges', 'expensive', 'long',
    'crowded', 'delayed', 'unreliable', 'unresponsive', 'confusing'
]

keyword_counts = {}
for keyword in complaint_keywords:
    count = all_negative_text.lower().count(keyword)
    if count > 0:
        keyword_counts[keyword] = count

print("Top complaint keywords in negative feedback:")
sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:10]
for keyword, count in sorted_keywords:
    print(f"  {keyword}: {count} mentions")