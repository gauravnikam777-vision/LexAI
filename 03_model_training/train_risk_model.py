"""
train_risk_model.py
===================
Step 3b – Train the Risk Level Classification Model
Author: Gaurav Govind Nikam

Model: TF-IDF + RandomForestClassifier
Task: Predict risk level (Low / Medium / High) from case description

Run: python train_risk_model.py
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

BASE      = os.path.dirname(os.path.dirname(__file__))
DATA_DIR  = os.path.join(BASE, 'data')
MODELS    = os.path.join(BASE, 'models')
os.makedirs(MODELS, exist_ok=True)


def train_risk_model():
    print("=" * 55)
    print("LexAI – Risk Level Classifier Training")
    print("=" * 55)

    # Load data
    json_path = os.path.join(DATA_DIR, 'advanced_cases.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.dropna(subset=['text', 'risk_level'])
    df = df[df['text'].str.len() > 20]
    print(f"Dataset: {len(df)} records")

    # Load pre-fitted vectorizer (same one as category model)
    vec_path = os.path.join(MODELS, 'vectorizer.pkl')
    if not os.path.exists(vec_path):
        print("[ERR] Run train_category_model.py first to create vectorizer.pkl")
        return None, None

    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    X = vectorizer.transform(df['text'])

    # Encode risk labels
    risk_enc = LabelEncoder()
    y = risk_enc.fit_transform(df['risk_level'])
    print(f"Risk levels: {list(risk_enc.classes_)}")

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    # NOTE: Risk prediction is harder than category — classes are more subjective
    # class_weight='balanced' is critical here — 'Low' risk cases are underrepresented
    print("\nTraining risk model...")
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc*100:.1f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=risk_enc.classes_, zero_division=0))

    # CV Score
    cv = cross_val_score(model, X, y, cv=5, n_jobs=-1)
    print(f"CV Accuracy: {cv.mean():.4f} ± {cv.std():.4f}")

    # Save
    with open(os.path.join(MODELS, 'risk_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(MODELS, 'risk_encoder.pkl'), 'wb') as f:
        pickle.dump(risk_enc, f)

    print("[OK] risk_model.pkl and risk_encoder.pkl saved")
    return model, risk_enc


if __name__ == '__main__':
    m, enc = train_risk_model()
    if m:
        print(f"\nNext step: python train_similarity_model.py")
