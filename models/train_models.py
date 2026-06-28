"""
train_models.py  –  LexAI Model Training
Handles 6 categories: fraud, property, labour, family, consumer, motor_accident, criminal
"""
import os, sys, pickle, json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import LabelEncoder

sys.stdout.reconfigure(encoding='utf-8')

DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'advanced_cases.json')
MODELS_DIR = os.path.dirname(__file__)

def save(obj, name):
    with open(os.path.join(MODELS_DIR, name), 'wb') as f:
        pickle.dump(obj, f)
    print(f"  [OK] Saved {name}")

def train():
    print("\n=== LexAI - Model Training ===\n")

    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data_json = json.load(f)

    df = pd.DataFrame(data_json)
    print(f"Loaded {len(df)} cases.")
    print(f"Categories:  {dict(df['category'].value_counts())}")
    print(f"Risk levels: {dict(df['risk_level'].value_counts())}")
    print(f"Outcomes:    {dict(df['outcome'].value_counts())}\n")

    texts      = df['text'].tolist()
    categories = df['category'].tolist()
    risks      = df['risk_level'].tolist()
    outcomes   = df['outcome'].tolist()

    # ── TF-IDF vectorizer ───────────────────────────────────────────────────
    print("Fitting TF-IDF Vectorizer ...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words='english',
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1
    )
    X = vectorizer.fit_transform(texts)
    save(vectorizer, 'vectorizer.pkl')

    # ── Category classifier ─────────────────────────────────────────────────
    print("\nTraining Category Classifier ...")
    cat_enc = LabelEncoder()
    y_cat = cat_enc.fit_transform(categories)
    cat_model = LogisticRegression(max_iter=3000, C=3.0, class_weight='balanced', random_state=42)
    cat_model.fit(X, y_cat)
    print(f"  Classes: {list(cat_enc.classes_)}")
    save(cat_model, 'category_model.pkl')
    save(cat_enc,   'category_encoder.pkl')

    # ── Risk predictor ──────────────────────────────────────────────────────
    print("Training Risk Predictor ...")
    risk_enc = LabelEncoder()
    y_risk = risk_enc.fit_transform(risks)
    risk_model = LogisticRegression(max_iter=3000, C=3.0, class_weight='balanced', random_state=42)
    risk_model.fit(X, y_risk)
    print(f"  Classes: {list(risk_enc.classes_)}")
    save(risk_model, 'risk_model.pkl')
    save(risk_enc,   'risk_encoder.pkl')

    # ── Outcome predictor ───────────────────────────────────────────────────
    # If dataset has only 1 outcome class, use DummyClassifier as placeholder.
    # Win probability is derived from risk level in app.py for accuracy.
    print("Training Outcome Predictor ...")
    out_enc = LabelEncoder()
    y_out = out_enc.fit_transform(outcomes)
    n_classes = len(np.unique(y_out))
    if n_classes < 2:
        print(f"  NOTE: Only 1 outcome class in dataset. Using DummyClassifier.")
        out_model = DummyClassifier(strategy='most_frequent')
    else:
        out_model = LogisticRegression(max_iter=3000, C=3.0, class_weight='balanced', random_state=42)
    out_model.fit(X, y_out)
    print(f"  Classes: {list(out_enc.classes_)}")
    save(out_model, 'outcome_model.pkl')
    save(out_enc,   'outcome_encoder.pkl')

    # ── Save TF-IDF matrix and records for RAG similarity search ───────────
    print("\nSaving TF-IDF matrix and case records ...")
    save(X,         'tfidf_matrix.pkl')
    save(texts,     'case_texts.pkl')
    save(data_json, 'case_records.pkl')

    print("\n=== All models saved successfully! ===")

if __name__ == '__main__':
    train()
