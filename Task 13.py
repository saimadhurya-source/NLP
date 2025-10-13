import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = {
    "text": [
        "I am so happy today",
        "This makes me angry",
        "I feel sad and down",
        "I am scared of darkness",
        "That surprise was awesome",
        "I am disgusted by this",
        "Everything feels normal"
    ],
    "emotion": [
        "joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"
    ]
}
df = pd.DataFrame(data)
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["emotion"], test_size=0.3, random_state=42)
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
model = LogisticRegression()
model.fit(X_train_vec, y_train)
pred = model.predict(X_test_vec)
print("\nModel Evaluation:\n")
print(classification_report(y_test, pred))
samples = [
    "I am very happy with the result",
    "This is making me so angry",
    "I feel scared about the future"
]
for s in samples:
    vec = vectorizer.transform([s])
    emotion = model.predict(vec)[0]
    print(f"\nText: {s}\nPredicted Emotion: {emotion}")
