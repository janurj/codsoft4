import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset (assuming it's in CSV format)
df = pd.read_csv(r"C:\intern\sms\spam.csv", encoding='latin-1')

# Drop any unnecessary columns that have no relevant data
df = df[['v1', 'v2']]

# Rename columns for convenience
df.columns = ['label', 'message']

# Encode the labels: spam = 1, ham = 0
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Print classification report for more details
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Save the model and vectorizer for future use
joblib.dump(model, 'sms_spam_classifier_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

# Function to predict if a given SMS is spam or not
def predict_sms(sms):
    # Load the saved model and vectorizer
    model = joblib.load('sms_spam_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')

    # Transform the input SMS using the vectorizer
    sms_tfidf = vectorizer.transform([sms])

    # Predict using the trained model
    prediction = model.predict(sms_tfidf)[0]

    # Output the result
    if prediction == 1:
        print("The SMS is classified as: Spam")
    else:
        print("The SMS is classified as: Ham")

# Example: Input SMS for testing
sms_input = input("Enter an SMS message to classify: ")
predict_sms(sms_input)
