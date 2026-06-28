"""
train_category_model.py
========================
Step 3a – Train the Legal Category Classification Model
Author: Gaurav Govind Nikam

Model: TF-IDF + RandomForestClassifier
Task: Predict legal category from case description text
Categories: fraud, property, labour, family, consumer, motor_accident, criminal

Run: python train_category_model.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE, 'data')
MODELS    = os.path.join(BASE, 'models')
os.makedirs(MODELS, exist_ok=True)


def load_data():
    """
    Data load karo — advanced_cases.json se.
    Pehle cleaned CSV try karo, phir raw JSON.
    """
    csv_path  = os.path.join(DATA_DIR, 'processed', 'cases_cleaned.csv')
    json_path = os.path.join(DATA_DIR, 'advanced_cases.json')

    if os.path.exists(csv_path):
        print(f"Loading cleaned CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        text_col = 'text_clean' if 'text_clean' in df.columns else 'text'
    elif os.path.exists(json_path):
        print(f"Cleaned CSV not found. Loading raw JSON: {json_path}")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        text_col = 'text'
    else:
        raise FileNotFoundError("No dataset found! Run data collection first.")

    print(f"Loaded {len(df)} records.")
    return df, text_col


def train_category_model():
    print("=" * 55)
    print("LexAI – Category Classifier Training")
    print("=" * 55)

    # 1. Load data
    df, text_col = load_data()

    # Filter valid rows
    df = df.dropna(subset=[text_col, 'category'])
    df = df[df[text_col].str.len() > 20]
    print(f"After filtering: {len(df)} records")

    # 2. Encode labels
    cat_enc = LabelEncoder()
    y = cat_enc.fit_transform(df['category'])
    print(f"Categories: {list(cat_enc.classes_)}")

    # 3. TF-IDF Vectorization
    # NOTE: Tried many values for max_features. 5000 gave best results.
    # Bigrams (ngram_range=(1,2)) help because "wrongful termination" is better than separate words
    print("\nBuilding TF-IDF vectors...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,   # apply log normalization — helps with imbalanced data
    )
    X = vectorizer.fit_transform(df[text_col])
    print(f"Feature matrix shape: {X.shape}")

    # 4. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # 5. Train Model
    # NOTE: First tried LogisticRegression — accuracy was 68%
    # Then tried SVM — good but slow for Flask serving
    # Settled on RandomForest — fast prediction + good accuracy (82%)
    # class_weight='balanced' is KEY — dataset thoda imbalanced tha
    print("\nTraining RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,      # tried 100 first, 200 gave +3% accuracy
        max_depth=None,        # let trees grow fully
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=cat_enc.classes_, zero_division=0))

    # 7. Cross-validation (5-fold)
    print("Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 8. Save models
    print("\nSaving models...")
    with open(os.path.join(MODELS, 'vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(MODELS, 'category_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS, 'category_encoder.pkl'), 'wb') as f:
        pickle.dump(cat_enc, f)

    # Also save TF-IDF matrix for similarity search
    tfidf_matrix = vectorizer.transform(df[text_col])
    with open(os.path.join(MODELS, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)

    print("[OK] Models saved to models/")
    print("  → vectorizer.pkl")
    print("  → category_model.pkl")
    print("  → category_encoder.pkl")
    print("  → tfidf_matrix.pkl")

    return model, vectorizer, cat_enc, acc


if __name__ == '__main__':
    model, vectorizer, enc, acc = train_category_model()

    # Quick sanity check
    print("\nSanity Check:")
    test_cases = [
        ("My boss fired me without notice and didn't pay salary", "labour"),
        ("Someone fraudulently transferred money from my bank account", "fraud"),
        ("My landlord is illegally trying to evict me", "property"),
    ]
    for text, expected in test_cases:
        vec = vectorizer.transform([text])
        pred_idx = model.predict(vec)[0]
        pred_cat = enc.inverse_transform([pred_idx])[0]
        match = "✓" if pred_cat == expected else "✗"
        print(f"  {match} Expected: {expected:<15} Got: {pred_cat}")

    print(f"\nFinal test accuracy: {acc*100:.1f}%")
    print("Next step: python train_risk_model.py")
