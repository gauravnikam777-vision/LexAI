"""
clean_text.py
=============
Step 2 – Text Cleaning and Preprocessing Pipeline
Author: Gaurav Govind Nikam

Ye script dataset ki text columns ko clean karta hai ML ke liye.
Indian legal text mein kuch special cases hain jo handle karne padte hain.

Run: python clean_text.py
"""

import re
import os
import json
import csv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_DIR = os.path.join(BASE, 'data')
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ------------------------------------------------------------------
# Custom Stopwords
# (NOTE: English NLTK stopwords remove kar dete the "not", "no" etc
#  jo legal text mein important hote hain. Isliye custom list banayi)
# ------------------------------------------------------------------
CUSTOM_STOPWORDS = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'has', 'have',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'shall', 'be', 'been', 'being', 'this', 'that',
    'these', 'those', 'they', 'their', 'there', 'then', 'than', 'to',
    'of', 'in', 'on', 'at', 'by', 'for', 'with', 'as', 'from',
    'it', 'its', 'into', 'through', 'and', 'or', 'but', 'so',
    # Hindi/Hinglish common words ko preserve kiya
    # 'karo', 'hai', 'mein' — ye intentionally NOT remove kiye
}

# Legal abbreviations expand karo (for better vectorization)
LEGAL_ABBREV = {
    r'\bipc\b':  'indian penal code',
    r'\bbns\b':  'bharatiya nyaya sanhita',
    r'\bcrpc\b': 'code of criminal procedure',
    r'\bbnss\b': 'bharatiya nagarik suraksha sanhita',
    r'\bfir\b':  'first information report',
    r'\brera\b': 'real estate regulatory authority',
    r'\bposh\b': 'prevention of sexual harassment',
    r'\bepf\b':  'employee provident fund',
    r'\bmact\b': 'motor accident claims tribunal',
    r'\bmlc\b':  'medico legal certificate',
    r'\brti\b':  'right to information',
}


def expand_abbreviations(text):
    """
    Legal abbreviations ko full form mein expand karo.
    Isse TF-IDF vectorizer better features banata hai.
    """
    text = text.lower()
    for pattern, replacement in LEGAL_ABBREV.items():
        text = re.sub(pattern, replacement, text)
    return text


def clean_text(text):
    """
    Main cleaning function.
    
    Steps:
    1. Lowercase
    2. Expand abbreviations
    3. Remove special chars (but keep numbers — IPC section numbers important)
    4. Remove extra spaces
    5. Remove stopwords
    """
    if not text or not isinstance(text, str):
        return ""

    # Step 1: lowercase
    text = text.lower()

    # Step 2: expand abbreviations
    text = expand_abbreviations(text)

    # Step 3: remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Step 4: remove emails
    text = re.sub(r'\S+@\S+', '', text)

    # Step 5: keep alphanumeric + spaces (preserve section numbers like "498a", "138")
    # NOTE: Tried removing all numbers first but IPC sections like "section 420" are KEY features
    text = re.sub(r'[^a-z0-9\s]', ' ', text)

    # Step 6: remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # Step 7: remove stopwords (custom list)
    words = text.split()
    words = [w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 1]

    return ' '.join(words)


def preprocess_dataset(input_path, output_path):
    """
    Full dataset ko clean karo aur CSV mein save karo.
    """
    print(f"Loading: {input_path}")

    # Try JSON first, then CSV
    if input_path.endswith('.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    else:
        df = pd.read_csv(input_path)

    print(f"Loaded {len(df)} records. Columns: {list(df.columns)}")

    # Required columns check
    if 'text' not in df.columns:
        print("[ERR] 'text' column not found!")
        return None

    # Clean text
    print("Cleaning text...")
    df['text_clean'] = df['text'].apply(clean_text)

    # Remove empty rows (after cleaning kuch rows empty ho jate hain)
    before = len(df)
    df = df[df['text_clean'].str.len() > 20]  # minimum 20 chars
    print(f"Removed {before - len(df)} rows with too-short text")

    # Fill missing labels
    if 'category' in df.columns:
        df['category'] = df['category'].fillna('criminal')
    if 'risk_level' in df.columns:
        df['risk_level'] = df['risk_level'].fillna('Medium')
    if 'outcome' in df.columns:
        df['outcome'] = df['outcome'].fillna('win')

    # Save cleaned CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"[SAVED] {output_path} ({len(df)} records)")

    return df


def encode_labels(df):
    """
    Category aur risk_level ko numbers mein convert karo.
    LabelEncoder fit karo aur pkl mein save karo.
    """
    print("\nEncoding labels...")

    # Category encoder
    cat_enc = LabelEncoder()
    df['category_enc'] = cat_enc.fit_transform(df['category'])
    print(f"  Categories: {list(cat_enc.classes_)}")

    # Risk encoder
    risk_enc = LabelEncoder()
    df['risk_enc'] = risk_enc.fit_transform(df['risk_level'])
    print(f"  Risk levels: {list(risk_enc.classes_)}")

    # Save encoders
    with open(os.path.join(MODELS_DIR, 'category_encoder.pkl'), 'wb') as f:
        pickle.dump(cat_enc, f)
    with open(os.path.join(MODELS_DIR, 'risk_encoder.pkl'), 'wb') as f:
        pickle.dump(risk_enc, f)

    print("  [SAVED] Encoders saved to models/")
    return df, cat_enc, risk_enc


if __name__ == '__main__':
    print("=" * 50)
    print("LexAI – Text Preprocessing Pipeline")
    print("=" * 50)

    # Input: advanced_cases.json (our main dataset)
    input_file = os.path.join(DATA_DIR, 'advanced_cases.json')
    output_file = os.path.join(DATA_DIR, 'processed', 'cases_cleaned.csv')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    df = preprocess_dataset(input_file, output_file)

    if df is not None and 'category' in df.columns:
        df, cat_enc, risk_enc = encode_labels(df)
        print(f"\nPreprocessing complete! {len(df)} records ready for training.")
        print(f"Next step: Run 03_model_training/train_category_model.py")
