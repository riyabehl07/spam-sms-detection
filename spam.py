import pandas as pd 
import numpy as np 
import re 
import string 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
nltk.download('punkt')

# Define the dataset path
dataset_path = r"C:\Users\Riya behl\Desktop\spam_sms_detection\archive (1)\spam.csv"

# Load dataset, keeping only required columns
df = pd.read_csv(dataset_path, encoding="latin-1")[['v1', 'v2']]
df.columns = ['label', 'message']  # Rename columns

# Convert labels to binary format (ham = 0, spam = 1)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Display dataset information
print(df.info())
print(df.head())

#Text Preprocessing
stop_words = set(stopwords.words("english"))  # Load stopwords

def clean_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    words = word_tokenize(text)  # Tokenize text into words
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return " ".join(words)  # Join words back into a cleaned sentence

df['cleaned_message'] = df['message'].apply(clean_text)  # Apply cleaning function

# Display cleaned text samples
print(df[['message', 'cleaned_message']].head())

#Convert Text into Numerical Features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Use top 5000 words
X = tfidf_vectorizer.fit_transform(df['cleaned_message'])  # Convert text to TF-IDF features

# Extract labels
y = df['label']

# Split dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

#Train Machine Learning Models

# Naïve Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

y_pred_nb = nb_model.predict(X_test)  # Predict on test set

print("Naïve Bayes Model Performance:")
print(accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))

# Logistic Regression
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

print("Logistic Regression Model Performance:")
print(accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Support Vector Machine (SVM)
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)

print("SVM Model Performance:")
print(accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm))

print(f"Naïve Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")

#save and load model
import joblib

joblib.dump(lr_model, "spam_classifier.pkl")  # Save Logistic Regression model
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")  # Save TF-IDF vectorizer
loaded_model = joblib.load("spam_classifier.pkl")
loaded_vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_spam(message):
    message_cleaned = clean_text(message)  # Clean text
    message_tfidf = loaded_vectorizer.transform([message_cleaned])  # Convert to TF-IDF
    prediction = loaded_model.predict(message_tfidf)[0]  # Predict
    return "Spam" if prediction == 1 else "Legitimate"

# Test predictions
print(predict_spam("You have won $1000! Click here to claim now."))
print(predict_spam("Hey, let's meet for lunch today."))





