# Phishing Email Detection Model (Mini Project)

A Scikit-learn model that classifies emails as **Phishing** or **Safe**
using a combination of TF-IDF text features and hand-engineered
signals (URLs, suspicious keywords, punctuation patterns, etc.).

## Files in this project

| File | Description |
|---|---|
| `phishing_detector.py` | Main script: feature engineering, training, evaluation, CLI |
| `dataset_generator.py` | Generates a synthetic (but realistic) labeled email dataset |
| `requirements.txt` | Python dependencies |
| `confusion_matrix.png` | Generated after training — visual confusion matrix |
| `emails_dataset.csv` | Generated dataset (created on first run) |
| `phishing_model.joblib` | Trained model, saved after training |

## Features

- **Text features**: TF-IDF vectorization (unigrams + bigrams) of the
  email subject + body.
- **Engineered features**:
  - Number of URLs in the email
  - Whether any URL uses a raw IP address instead of a domain
  - Whether a URL shortener (bit.ly, tinyurl, etc.) is used
  - Whether a suspicious top-level domain is present (`.tk`, `.xyz`, etc.)
  - Count of urgency/scare-tactic keywords ("urgent", "verify",
    "suspended", "act now", "claim", "refund", etc.)
  - Number of exclamation marks and all-caps words
  - Email length
  - Presence of generic greetings like "Dear Customer" / "Dear User"
- **Model**: Logistic Regression by default (interpretable, fast),
  with an optional Random Forest (`--model random_forest`).
- **Feature scaling**: engineered numeric features are standardized
  before being combined with TF-IDF — this matters because raw values
  like `text_length` (~100s) can otherwise dominate over TF-IDF values
  (~0–1) and produce an unstable, overconfident model.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Train on a synthetic dataset (auto-generated) and evaluate
python3 phishing_detector.py

# Train using Random Forest instead of Logistic Regression
python3 phishing_detector.py --model random_forest

# Use your own dataset (must have 'text' and 'label' columns, label: 1=phishing, 0=safe)
python3 phishing_detector.py --data my_emails.csv

# Train, then classify a single email you provide
python3 phishing_detector.py --classify "Urgent! Verify your account now at http://192.168.1.5/login"
```

## Output

Each run prints:
- Accuracy, Precision, Recall, F1-score
- A confusion matrix (text + saved as `confusion_matrix.png`)
- Full classification report
- 5-fold cross-validation accuracy (sanity check against overfitting)
- Top features pushing predictions toward "Phishing" (for Logistic Regression)

### Sample result (synthetic dataset, 600 emails, 75/25 split)

```
Accuracy  : 1.0000
Precision : 1.0000
Recall    : 1.0000
F1-score  : 1.0000
```

Note: near-perfect accuracy here reflects the synthetic dataset's
templated structure (real-world phishing emails are messier). For a
more rigorous evaluation, swap in a real dataset — e.g. Kaggle's
"Phishing Email Dataset" or the Nazario phishing corpus — via `--data`.

## Using a real dataset

Any CSV with these two columns works:

```csv
text,label
"Urgent: verify your account now at http://...",1
"Hi, please find attached the invoice for August.",0
```

## How it works (for the project write-up)

1. **Data**: Since public phishing-email datasets vary widely in
   format, this project includes a synthetic generator that mixes
   realistic phishing templates (urgency, prize claims, account
   suspension, fake security alerts) with legitimate business/personal
   email templates, so the pipeline is fully reproducible without
   external downloads.
2. **Feature extraction**: Raw text alone often isn't the strongest
   phishing signal — URL structure and manipulation tactics matter
   just as much, so both are combined into one feature matrix.
3. **Scaling**: TF-IDF values and engineered counts live on very
   different numeric scales. Standardizing the engineered features
   before combining prevents the linear model from being thrown off
   by scale differences (verified during testing — an earlier
   unscaled version misclassified some inputs with high confidence).
4. **Model**: Logistic Regression gives interpretable coefficients
   (useful for explaining *why* an email was flagged); Random Forest
   is offered as a non-linear alternative.
5. **Evaluation**: accuracy alone can be misleading for security
   classifiers, so precision, recall, F1, a full confusion matrix, and
   cross-validation are all reported.

## Possible extensions

- Train on a real-world labeled dataset (e.g. Enron + phishing corpora)
- Add sender-domain / SPF-DKIM-based features
- Add a simple web UI (Flask/Streamlit) for pasting in an email to classify
- Try deep learning (e.g. a fine-tuned DistilBERT) for comparison
- Track false-negative rate specifically, since missed phishing is costlier than false alarms
