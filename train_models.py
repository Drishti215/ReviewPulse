import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import joblib

# Example dataset (replace with your own reviews + labels)
data = {
    "review": [
        "This movie was amazing, I loved it!",
        "The plot was boring and the acting was terrible.",
        "Fantastic experience, highly recommend!",
        "Worst film I have ever seen."
    ],
    "label": [1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}
df = pd.DataFrame(data)

# Vectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["review"])
y = df["label"]

# Train Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X, y)

# Train Naive Bayes
naive_model = MultinomialNB()
naive_model.fit(X, y)

# Save models + vectorizer
joblib.dump(logistic_model, "logistic_model.pkl")
joblib.dump(naive_model, "naive_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(" Models and vectorizer saved successfully!")
