import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_excel('customer-churn-data_usecase2_Hackathon.xlsx')

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\n" + "="*50)
print("DATASET OVERVIEW")
print("="*50)
print("\nColumn Names and Data Types:")
print(df.dtypes)
print("\n" + "="*50)
print("First 5 rows:")
print(df.head())

print("\n" + "="*50)
print("Dataset Info:")
df.info()