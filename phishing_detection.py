import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Sample email dataset
data = {
    "email": [
        "Congratulations! You won a $1000 prize. Click here to claim now.",
        "Your account has been suspended. Verify your password immediately.",
        "You have received a free gift. Click the link to claim your reward.",
        "Urgent! Your bank account will be blocked. Confirm your details now.",
        "Click here to win a free iPhone.",
        "Your payment was successful. Thank you for your purchase.",
        "Meeting is scheduled for tomorrow at 10 AM.",
        "Please find the project report attached.",
        "Your Amazon order has been shipped.",
        "Can we meet today to discuss the project?",
        "The college assignment submission deadline is tomorrow.",
        "Your interview is scheduled for Monday."
    ],
    "label": [
        "Phishing",
        "Phishing",
        "Phishing",
        "Phishing",
        "Phishing",
        "Legitimate",
        "Legitimate",
        "Legitimate",
        "Legitimate",
        "Legitimate",
        "Legitimate",
        "Legitimate"
    ]
}

# Convert data into DataFrame
df = pd.DataFrame(data)

# Separate emails and labels
X = df["email"]
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Convert text into numerical features
vectorizer = TfidfVectorizer()

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Create machine learning model
model = LogisticRegression()

# Train model
model.fit(X_train_vectorized, y_train)

# Test model
y_pred = model.predict(X_test_vectorized)

# Display accuracy
accuracy = accuracy_score(y_test, y_pred)

print("====================================")
print("   PHISHING EMAIL DETECTION SYSTEM")
print("====================================")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Function for predicting a new email
def check_email(email):
    email_vectorized = vectorizer.transform([email])
    prediction = model.predict(email_vectorized)[0]

    print("\nEmail:")
    print(email)
    print("\nPrediction:", prediction)


# Test emails
check_email(
    "URGENT! Your bank account will be closed. Click this link and verify your password."
)

check_email(
    "Hello, the project meeting is scheduled for tomorrow at 10 AM."
)