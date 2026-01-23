# %%
import pandas as pd
import re
import nltk
import joblib

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# %%

nltk.download('stopwords')


# %%
data = pd.read_csv("spam_ham_dataset.csv")
data = data[['text', 'label_num']]
data = data.dropna()

print(data.head())
print(data.shape)

# %%
def clean_email(text):
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    words = text.split()
    words = [w for w in words if w not in stopwords.words('english')]

    return " ".join(words)

# %%
data['clean_text'] = data['text'].apply(clean_email)

print(data[['text', 'clean_text']].head())

# %%
vectorizer = TfidfVectorizer(
    max_features=6000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(data['clean_text'])
y = data['label_num']

print(X.shape)

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)

# %%
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training complete!")


# %%
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# %%
def predict_email(email):
    email = clean_email(email)
    vector = vectorizer.transform([email])
    prediction = model.predict(vector)
    return " spam email " if prediction[0] == 1 else " legit email "

# %%
print(predict_email("Congratulations! You have won $10,000. Click here now!"))
print(predict_email("Hey, are we still meeting for the project tomorrow?"))

# %%
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("model and vectorizer saved successfully.")

