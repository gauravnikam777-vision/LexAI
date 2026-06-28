"""
retrain_models.py
=================
Retrain ML models using LOCAL sklearn version (1.5.1) to fix version mismatch.
The old .pkl files were trained on sklearn 1.8.0 which caused predict_proba errors.

Run: python retrain_models.py
"""

import os, json, pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

BASE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE, 'data')
MODELS    = os.path.join(BASE, 'models')
os.makedirs(MODELS, exist_ok=True)

print("=" * 55)
print("LexAI — Model Retraining (sklearn compatibility fix)")
print("=" * 55)

# ────────────────────────────────────────────────
# 1. Load dataset
# ────────────────────────────────────────────────
json_path = os.path.join(DATA_DIR, 'advanced_cases.json')
if not os.path.exists(json_path):
    print(f"[ERR] Dataset not found: {json_path}")
    import sys; sys.exit(1)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Dataset loaded: {len(data)} records")

texts      = [d['text'] for d in data]
categories = [d['category'] for d in data]
risks      = [d['risk_level'] for d in data]

import pandas as pd
df = pd.DataFrame({'text': texts, 'category': categories, 'risk_level': risks})
df = df.dropna()
df = df[df['text'].str.len() > 10]
print(f"After cleanup: {len(df)} records")
print(f"Categories: {df['category'].value_counts().to_dict()}")
print(f"Risks:      {df['risk_level'].value_counts().to_dict()}")

# ────────────────────────────────────────────────
# 2. TF-IDF Vectorizer
# ────────────────────────────────────────────────
print("\nBuilding TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features   = 5000,
    ngram_range    = (1, 2),
    sublinear_tf   = True,
    min_df         = 1,
)
X = vectorizer.fit_transform(df['text'])
print(f"Vocab size: {len(vectorizer.vocabulary_)}")

# ────────────────────────────────────────────────
# 3. Category Classifier (RandomForest)
# ────────────────────────────────────────────────
print("\nTraining category classifier...")
cat_enc = LabelEncoder()
y_cat   = cat_enc.fit_transform(df['category'])
print(f"Category classes: {list(cat_enc.classes_)}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y_cat, test_size=0.2, random_state=42)
cat_model = RandomForestClassifier(
    n_estimators   = 200,
    class_weight   = 'balanced',
    random_state   = 42,
    n_jobs         = -1,
)
cat_model.fit(X_tr, y_tr)
y_pred = cat_model.predict(X_te)
acc_cat = accuracy_score(y_te, y_pred)
print(f"Category Accuracy: {acc_cat*100:.1f}%")
print(classification_report(y_te, y_pred, labels=range(len(cat_enc.classes_)), target_names=cat_enc.classes_, zero_division=0))

# Test predict_proba works
cat_test_proba = cat_model.predict_proba(X_te[:1])
print(f"predict_proba shape: {cat_test_proba.shape} — OK!")

# ────────────────────────────────────────────────
# 4. Risk Level Classifier (RandomForest)
# ────────────────────────────────────────────────
print("\nTraining risk classifier...")
risk_enc = LabelEncoder()
y_risk   = risk_enc.fit_transform(df['risk_level'])
print(f"Risk classes: {list(risk_enc.classes_)}")

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y_risk, test_size=0.2, random_state=42)
risk_model = RandomForestClassifier(
    n_estimators   = 200,
    class_weight   = 'balanced',
    random_state   = 42,
    n_jobs         = -1,
)
risk_model.fit(X_tr2, y_tr2)
y_pred2 = risk_model.predict(X_te2)
acc_risk = accuracy_score(y_te2, y_pred2)
print(f"Risk Accuracy: {acc_risk*100:.1f}%")
print(classification_report(y_te2, y_pred2, labels=range(len(risk_enc.classes_)), target_names=risk_enc.classes_, zero_division=0))

# Test predict_proba
risk_test_proba = risk_model.predict_proba(X_te2[:1])
print(f"predict_proba shape: {risk_test_proba.shape} — OK!")

# ────────────────────────────────────────────────
# 5. TF-IDF Matrix for similarity search
# ────────────────────────────────────────────────
print("\nBuilding similarity index (all records)...")
tfidf_matrix = vectorizer.transform(df['text'].tolist())

# Build case records (for RAG display)
case_records = []
for i, d in enumerate(data):
    # how_to_do — try to parse as JSON list
    how_raw = d.get('how_to_do', '[]')
    if isinstance(how_raw, list):
        how_list = how_raw
    elif isinstance(how_raw, str):
        try:
            how_list = json.loads(how_raw)
        except Exception:
            how_list = [how_raw] if how_raw.strip() else []
    else:
        how_list = []

    case_records.append({
        'case_id'     : d.get('case_id', f'case_{i+1}'),
        'text'        : d.get('text', '')[:300],
        'category'    : d.get('category', ''),
        'risk_level'  : d.get('risk_level', 'Medium'),
        'outcome'     : d.get('outcome', 'win'),
        'ipc_sections': d.get('ipc_sections', ''),
        'what_to_do'  : d.get('what_to_do', ''),
        'how_to_do'   : how_list,
        'when_to_do'  : d.get('when_to_do', ''),
    })

print(f"Case records built: {len(case_records)}")
if case_records:
    print(f"Sample: {case_records[0].get('what_to_do','')[:60]}")
    print(f"  how_to_do items: {len(case_records[0].get('how_to_do',[]))}")

# ────────────────────────────────────────────────
# 6. Save all models
# ────────────────────────────────────────────────
print("\nSaving models...")
saves = {
    'vectorizer.pkl'     : vectorizer,
    'category_model.pkl' : cat_model,
    'category_encoder.pkl': cat_enc,
    'risk_model.pkl'     : risk_model,
    'risk_encoder.pkl'   : risk_enc,
    'tfidf_matrix.pkl'   : tfidf_matrix,
    'case_records.pkl'   : case_records,
}
for fname, obj in saves.items():
    with open(os.path.join(MODELS, fname), 'wb') as f:
        pickle.dump(obj, f)
    print(f"  [saved] {fname}")

# ────────────────────────────────────────────────
# 7. Quick sanity test
# ────────────────────────────────────────────────
print("\n=== Sanity Test ===")
test_cases = [
    "My boss fired me without notice and did not pay my last 3 months salary",
    "Cheque bounce ho gaya, kya karna chahiye",
    "498A case filed against me, wife left home",
    "UPI fraud ho gaya, paise chale gaye",
    "Builder ne flat nahi diya, RERA complaint karna hai",
]
for tc in test_cases:
    v   = vectorizer.transform([tc])
    ci  = cat_model.predict(v)[0]
    cat = cat_enc.inverse_transform([ci])[0]
    cp  = cat_model.predict_proba(v)[0].max()
    ri  = risk_model.predict(v)[0]
    rsk = risk_enc.inverse_transform([ri])[0]
    rp  = risk_model.predict_proba(v)[0].max()
    print(f"  '{tc[:50]}...' -> {cat}({cp*100:.0f}%) | {rsk}({rp*100:.0f}%)")

print(f"\nAll done! Models saved to: {MODELS}")
print("Restart the Flask server: python app.py")
