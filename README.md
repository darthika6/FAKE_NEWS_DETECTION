# FAKE_NEWS_DETECTION
# ============================================================
#        FAKE NEWS DETECTION USING MACHINE LEARNING
#              COMPLETE GOOGLE COLAB CODE
#                    ONE CELL ONLY
# ============================================================

# -------------------- 1. INSTALL ----------------------------

!pip install -q pandas numpy scikit-learn matplotlib seaborn


# -------------------- 2. IMPORT LIBRARIES --------------------

import os
import zipfile
import re
import string
import pickle

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import files

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 3. UPLOAD FAKE.CSV.ZIP
# ============================================================

print("==============================================")
print("STEP 1: Upload Fake.csv.zip")
print("==============================================")

fake_uploaded = files.upload()

fake_zip = None

for file_name in fake_uploaded.keys():
    if file_name.lower().endswith(".zip"):
        fake_zip = file_name
        break

if fake_zip is None:
    raise Exception("Please upload Fake.csv.zip")


print("\nFake dataset uploaded:", fake_zip)


# ============================================================
# 4. EXTRACT FAKE.CSV
# ============================================================

with zipfile.ZipFile(fake_zip, "r") as zip_ref:
    zip_ref.extractall("/content")

print("Fake.csv extracted successfully!")


# ============================================================
# 5. UPLOAD TRUE.CSV.ZIP
# ============================================================

print("\n==============================================")
print("STEP 2: Upload True.csv.zip")
print("==============================================")

true_uploaded = files.upload()

true_zip = None

for file_name in true_uploaded.keys():
    if file_name.lower().endswith(".zip"):
        true_zip = file_name
        break

if true_zip is None:
    raise Exception("Please upload True.csv.zip")


print("\nTrue dataset uploaded:", true_zip)


# ============================================================
# 6. EXTRACT TRUE.CSV
# ============================================================

with zipfile.ZipFile(true_zip, "r") as zip_ref:
    zip_ref.extractall("/content")

print("True.csv extracted successfully!")


# ============================================================
# 7. CHECK FILES
# ============================================================

if not os.path.exists("/content/Fake.csv"):
    raise FileNotFoundError("Fake.csv not found!")

if not os.path.exists("/content/True.csv"):
    raise FileNotFoundError("True.csv not found!")

print("\n==============================================")
print("Both datasets are ready!")
print("==============================================")


# ============================================================
# 8. LOAD DATASET
# ============================================================

fake = pd.read_csv("/content/Fake.csv")
true = pd.read_csv("/content/True.csv")

print("\nFake News records :", len(fake))
print("Real News records :", len(true))


# ============================================================
# 9. CHECK COLUMNS
# ============================================================

print("\nFake.csv columns:")
print(fake.columns.tolist())

print("\nTrue.csv columns:")
print(true.columns.tolist())


# ============================================================
# 10. SELECT TITLE AND TEXT
# ============================================================

if "title" not in fake.columns or "text" not in fake.columns:
    raise Exception(
        "Fake.csv must contain 'title' and 'text' columns."
    )

if "title" not in true.columns or "text" not in true.columns:
    raise Exception(
        "True.csv must contain 'title' and 'text' columns."
    )

fake = fake[["title", "text"]].copy()
true = true[["title", "text"]].copy()


# ============================================================
# 11. ADD LABELS
# ============================================================

# 0 = Fake News
# 1 = Real News

fake["label"] = 0
true["label"] = 1


# ============================================================
# 12. COMBINE DATASETS
# ============================================================

data = pd.concat(
    [fake, true],
    ignore_index=True
)

# Remove missing values
data = data.dropna()

# Remove duplicates
data = data.drop_duplicates()

# Shuffle dataset
data = data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nFinal dataset size:", data.shape)


# ============================================================
# 13. COMBINE TITLE + TEXT
# ============================================================

data["content"] = (
    data["title"].astype(str)
    + " "
    + data["text"].astype(str)
)


# ============================================================
# 14. TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML tags
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


print("\nCleaning text...")

data["content"] = data["content"].apply(clean_text)

print("Text cleaning completed!")


# ============================================================
# 15. INPUT AND OUTPUT
# ============================================================

X = data["content"]
y = data["label"]


# ============================================================
# 16. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# ============================================================
# 17. TF-IDF
# ============================================================

print("\nConverting text into numbers using TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=50000,
    ngram_range=(1, 2)
)

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF completed!")


# ============================================================
# 18. MACHINE LEARNING MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train_tfidf,
    y_train
)

print("Model training completed!")


# ============================================================
# 19. PREDICTION ON TEST DATA
# ============================================================

y_pred = model.predict(X_test_tfidf)


# ============================================================
# 20. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n")
print("==============================================")
print("             MODEL PERFORMANCE")
print("==============================================")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


# ============================================================
# 21. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")
print("----------------------------------------------")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Fake News",
            "Real News"
        ]
    )
)


# ============================================================
# 22. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Fake",
        "Real"
    ],
    yticklabels=[
        "Fake",
        "Real"
    ]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fake News Detection - Confusion Matrix")

plt.show()


# ============================================================
# 23. PREDICTION FUNCTION
# ============================================================

def predict_news(news):

    # Clean the news
    cleaned_news = clean_text(news)

    # Convert to TF-IDF
    news_vector = vectorizer.transform(
        [cleaned_news]
    )

    # Prediction
    prediction = model.predict(
        news_vector
    )[0]

    # Probability
    probability = model.predict_proba(
        news_vector
    )[0]

    print("\n")
    print("==============================================")
    print("                 RESULT")
    print("==============================================")

    if prediction == 0:

        print("Prediction : FAKE NEWS")
        print(
            "Confidence :",
            round(
                probability[0] * 100,
                2
            ),
            "%"
        )

    else:

        print("Prediction : REAL NEWS")
        print(
            "Confidence :",
            round(
                probability[1] * 100,
                2
            ),
            "%"
        )


# ============================================================
# 24. TEST YOUR OWN NEWS
# ============================================================

print("\n")
print("==============================================")
print("             TEST YOUR OWN NEWS")
print("==============================================")

news = input(
    "Enter your news article: "
)

predict_news(news)


# ============================================================
# 25. SAVE MODEL
# ============================================================

with open(
    "/content/fake_news_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


with open(
    "/content/tfidf_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f
    )


# ============================================================
# 26. DOWNLOAD MODEL FILES
# ============================================================

print("\n")
print("==============================================")
print("             PROJECT COMPLETED")
print("==============================================")

print("Model saved successfully!")
print("Vectorizer saved successfully!")

print("\nDownloading model files...")

files.download(
    "/content/fake_news_model.pkl"
)

files.download(
    "/content/tfidf_vectorizer.pkl"
)

print("\nDONE! Your Fake News Detection project is completed.")
