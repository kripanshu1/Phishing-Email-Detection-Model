#!/usr/bin/env python3
"""
Phishing Email Detection Model (Mini Project)
===============================================

Trains a Scikit-learn model to classify emails as "Phishing" or
"Safe" using a combination of:
  1. TF-IDF text features from the email subject + body
  2. Hand-engineered features (URL patterns, suspicious keywords,
     punctuation/urgency signals, etc.)

Outputs:
  - Console: accuracy, precision/recall/F1, confusion matrix
  - confusion_matrix.png : visual confusion matrix
  - phishing_model.joblib : the trained pipeline, ready for reuse

Usage
-----
    python3 phishing_detector.py                     # generate data, train, evaluate
    python3 phishing_detector.py --data my_emails.csv  # use your own CSV (text,label)
    python3 phishing_detector.py --classify "Urgent! Verify your account at http://..."
"""

import argparse
import re
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from dataset_generator import generate_dataset


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------

SUSPICIOUS_KEYWORDS = [
    "urgent", "verify", "suspend", "suspended", "account", "click here",
    "immediately", "password", "bank", "prize", "winner", "congratulations",
    "act now", "limited", "confirm", "security alert", "update your",
    "claim", "refund", "free", "gift card", "expire", "final notice",
    "unusual activity", "reset your", "credentials",
]

URL_PATTERN = re.compile(r"https?://[^\s]+")
IP_URL_PATTERN = re.compile(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
SHORTENER_PATTERN = re.compile(
    r"(bit\.ly|tinyurl\.com|goo\.gl|t\.co|ow\.ly|is\.gd|buff\.ly)", re.IGNORECASE)
SUSPICIOUS_TLD_PATTERN = re.compile(r"\.(tk|xyz|info|top|club|gq|ml)(/|$)", re.IGNORECASE)


def extract_engineered_features(texts):
    """Given a list/Series of raw email text, return a DataFrame of
    hand-crafted numeric features useful for phishing detection."""
    rows = []
    for text in texts:
        text_lower = text.lower()
        urls = URL_PATTERN.findall(text)

        rows.append({
            "num_urls": len(urls),
            "has_ip_url": int(bool(IP_URL_PATTERN.search(text))),
            "has_shortened_url": int(bool(SHORTENER_PATTERN.search(text))),
            "has_suspicious_tld": int(bool(SUSPICIOUS_TLD_PATTERN.search(text))),
            "num_exclamations": text.count("!"),
            "num_suspicious_keywords": sum(
                1 for kw in SUSPICIOUS_KEYWORDS if kw in text_lower),
            "text_length": len(text),
            "num_uppercase_words": sum(
                1 for w in text.split() if w.isupper() and len(w) > 2),
            "has_dear_customer": int(
                "dear customer" in text_lower or "dear user" in text_lower),
        })
    return pd.DataFrame(rows)


class PhishingFeatureExtractor:
    """Combines TF-IDF text vectorization with engineered numeric
    features into a single feature matrix for the classifier."""

    def __init__(self, max_tfidf_features=1500):
        self.vectorizer = TfidfVectorizer(
            max_features=max_tfidf_features, stop_words="english", ngram_range=(1, 2))
        # Engineered numeric features live on very different scales than
        # TF-IDF values (e.g. text_length ~100s vs TF-IDF ~0-1). Without
        # scaling, a linear model can develop unstable, overconfident
        # coefficients -- especially on separable data -- so we standardize
        # them before combining with the sparse text matrix.
        self.scaler = StandardScaler()
        self.feature_names_ = None

    def fit_transform(self, texts):
        tfidf = self.vectorizer.fit_transform(texts)
        engineered = extract_engineered_features(texts)
        self.engineered_columns_ = engineered.columns.tolist()
        engineered_scaled = self.scaler.fit_transform(engineered.values)
        combined = hstack([tfidf, csr_matrix(engineered_scaled)])
        self.feature_names_ = (
            list(self.vectorizer.get_feature_names_out()) + self.engineered_columns_)
        return combined

    def transform(self, texts):
        tfidf = self.vectorizer.transform(texts)
        engineered = extract_engineered_features(texts)
        engineered = engineered[self.engineered_columns_]  # keep column order
        engineered_scaled = self.scaler.transform(engineered.values)
        combined = hstack([tfidf, csr_matrix(engineered_scaled)])
        return combined


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_dataset(path=None):
    if path:
        df = pd.read_csv(path)
        if "text" not in df.columns or "label" not in df.columns:
            print("[!] CSV must contain 'text' and 'label' columns.")
            sys.exit(1)
        print(f"[+] Loaded {len(df)} emails from {path}")
    else:
        print("[+] No dataset provided -- generating a synthetic dataset "
              "of phishing + legitimate emails for training/demo purposes.")
        df = generate_dataset(n_phishing=300, n_legit=300)
        df.to_csv("emails_dataset.csv", index=False)
        print(f"[+] Generated {len(df)} emails -> emails_dataset.csv")
    return df


# --------------------------------------------------------------------------
# Training & evaluation
# --------------------------------------------------------------------------

def train_and_evaluate(df, model_type="logistic", test_size=0.25):
    X_text = df["text"].tolist()
    y = df["label"].values

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=42, stratify=y)

    extractor = PhishingFeatureExtractor()
    X_train = extractor.fit_transform(X_train_text)
    X_test = extractor.transform(X_test_text)

    if model_type == "random_forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        # C=0.5 adds a bit more regularization than the default (1.0) to
        # keep coefficients from growing too extreme on easily-separable
        # data, which otherwise makes the model overconfident and brittle
        # on inputs that look slightly different from the training set.
        clf = LogisticRegression(max_iter=1000, C=0.5)

    print(f"[+] Training {clf.__class__.__name__} on {X_train.shape[0]} emails "
          f"({X_train.shape[1]} features)...")

    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    print(f"[+] 5-fold cross-validation accuracy on training data: "
          f"{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print(" MODEL EVALUATION")
    print("=" * 60)
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"                 Predicted Safe   Predicted Phishing")
    print(f"Actual Safe      {cm[0][0]:<16} {cm[0][1]}")
    print(f"Actual Phishing  {cm[1][0]:<16} {cm[1][1]}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, target_names=["Safe", "Phishing"]))

    plot_confusion_matrix(cm, out_path="confusion_matrix.png")

    if model_type == "logistic":
        show_top_features(clf, extractor)

    return clf, extractor, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(cm, out_path="confusion_matrix.png"):
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Safe", "Phishing"],
                yticklabels=["Safe", "Phishing"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Phishing Email Detection")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[+] Confusion matrix saved -> {out_path}")


def show_top_features(clf, extractor, top_n=10):
    """For logistic regression, show which features push most strongly
    toward 'Phishing'."""
    coefs = clf.coef_[0]
    names = extractor.feature_names_
    top_idx = np.argsort(coefs)[-top_n:][::-1]
    print(f"\nTop {top_n} features indicating PHISHING:")
    for i in top_idx:
        print(f"  {names[i]:<25} weight={coefs[i]:.3f}")


# --------------------------------------------------------------------------
# Single-email classification
# --------------------------------------------------------------------------

def classify_email(clf, extractor, text):
    X = extractor.transform([text])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)[0]
    label = "Phishing" if pred == 1 else "Safe"
    confidence = proba[pred]
    return label, confidence


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Phishing Email Detection Model (Scikit-learn mini project)")
    parser.add_argument("--data", default=None,
                         help="Path to a CSV with 'text' and 'label' columns. "
                              "If omitted, a synthetic dataset is generated.")
    parser.add_argument("--model", choices=["logistic", "random_forest"],
                         default="logistic", help="Classifier to use (default: logistic)")
    parser.add_argument("--classify", default=None,
                         help="After training, classify a single email text you provide.")
    parser.add_argument("--save-model", default="phishing_model.joblib",
                         help="Path to save the trained model+extractor.")
    args = parser.parse_args()

    print("=" * 60)
    print(" Phishing Email Detection Model - Mini Project")
    print("=" * 60)

    df = load_dataset(args.data)
    print(f"[+] Dataset balance -> Phishing: {sum(df['label'] == 1)}, "
          f"Safe: {sum(df['label'] == 0)}")

    clf, extractor, metrics = train_and_evaluate(df, model_type=args.model)

    joblib.dump({"model": clf, "extractor": extractor}, args.save_model)
    print(f"\n[+] Trained model saved -> {args.save_model}")

    if args.classify:
        label, confidence = classify_email(clf, extractor, args.classify)
        print("\n" + "-" * 60)
        print(f"Input email : {args.classify[:100]}...")
        print(f"Prediction  : {label}  (confidence: {confidence:.2%})")
        print("-" * 60)


if __name__ == "__main__":
    main()
