"""
train_similarity_model.py
=========================
Step 3c – Build TF-IDF Similarity Matrix for Case Retrieval (RAG)
Author: Gaurav Govind Nikam

This is NOT a traditional ML model.
It's a Retrieval-Augmented Generation (RAG) setup using TF-IDF cosine similarity.

When user submits a case, we find the TOP-3 most similar past cases from our dataset
and show them as reference. This gives real-world context to the AI's response.

Run: python train_similarity_model.py
"""

import os
import pickle
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

BASE     = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE, 'data')
MODELS   = os.path.join(BASE, 'models')


def build_similarity_index():
    print("=" * 55)
    print("LexAI – Similarity Index Builder (RAG)")
    print("=" * 55)

    # Load dataset
    json_path = os.path.join(DATA_DIR, 'advanced_cases.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} cases from dataset")

    # Load vectorizer (must be same as category model)
    vec_path = os.path.join(MODELS, 'vectorizer.pkl')
    if not os.path.exists(vec_path):
        print("[ERR] vectorizer.pkl not found. Run train_category_model.py first.")
        return

    with open(vec_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Build TF-IDF matrix for all cases
    texts = [d.get('text','') for d in data]
    print(f"Building TF-IDF matrix for {len(texts)} documents...")
    tfidf_matrix = vectorizer.transform(texts)
    print(f"Matrix shape: {tfidf_matrix.shape}")

    # Save matrix
    with open(os.path.join(MODELS, 'tfidf_matrix.pkl'), 'wb') as f:
        pickle.dump(tfidf_matrix, f)
    print(f"[OK] tfidf_matrix.pkl saved")

    # Save case records (for displaying similar cases)
    case_records = []
    for rec in data:
        case_records.append({
            'case_id'   : rec.get('case_id', ''),
            'text'      : rec.get('text', '')[:300],
            'category'  : rec.get('category', ''),
            'risk_level': rec.get('risk_level', ''),
            'outcome'   : rec.get('outcome', 'win'),
            'ipc_sections': rec.get('ipc_sections', ''),
            'what_to_do': rec.get('what_to_do', ''),
            'how_to_do' : rec.get('how_to_do', '[]'),
            'when_to_do': rec.get('when_to_do', ''),
        })

    with open(os.path.join(MODELS, 'case_records.pkl'), 'wb') as f:
        pickle.dump(case_records, f)
    print(f"[OK] case_records.pkl saved ({len(case_records)} records)")

    # Quick similarity test
    print("\nSanity check — similarity test:")
    test_query = "My employer fired me without notice and did not pay salary"
    test_vec = vectorizer.transform([test_query])
    sims = cosine_similarity(test_vec, tfidf_matrix).flatten()
    top3 = sims.argsort()[-3:][::-1]
    print(f"  Query: '{test_query}'")
    for idx in top3:
        if sims[idx] > 0.05:
            rec = case_records[idx]
            print(f"  → [{sims[idx]*100:.1f}%] [{rec['category']}] {rec['text'][:80]}…")

    print("\nSimilarity index built successfully.")
    print("All models are ready. Run model_evaluation.py to check accuracy.")


if __name__ == '__main__':
    build_similarity_index()
