"""
merge_datasets.py
=================
Step 1b – Merge multiple data sources into one final dataset
Author: Gaurav Govind Nikam

Humne multiple sources se data collect kiya:
1. Indian Kanoon (jo mila)
2. Manually written cases
3. Generated cases (generate_law_dataset.py se)

Is script mein sab ko merge karke ek single file banate hain.
"""

import json
import csv
import os
import random

BASE = os.path.dirname(os.path.dirname(__file__))  # project root
DATA_DIR = os.path.join(BASE, 'data')

def load_json_file(path):
    if not os.path.exists(path):
        print(f"  [SKIP] File not found: {path}")
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"  [OK] Loaded {len(data)} records from {os.path.basename(path)}")
    return data

def deduplicate(cases):
    """
    Duplicate cases remove karo based on case_id.
    Same text ke cases bhi remove karo (similarity threshold).
    """
    seen_ids = set()
    seen_texts = set()
    unique = []

    for case in cases:
        cid = case.get('case_id', '')
        text = case.get('text', '').strip()[:100]  # first 100 chars as fingerprint

        if cid in seen_ids or text in seen_texts:
            continue

        seen_ids.add(cid)
        seen_texts.add(text)
        unique.append(case)

    print(f"  [INFO] Dedup: {len(cases)} -> {len(unique)} unique records")
    return unique

def check_class_balance(cases):
    """
    Dataset mein kitne cases each category ke hain check karo.
    Imbalanced dataset model ko biased bana deta hai.
    """
    from collections import Counter
    cats = Counter(c.get('category', 'unknown') for c in cases)
    print("\n  Category Distribution:")
    for cat, count in sorted(cats.items()):
        bar = '█' * (count // 2)
        print(f"    {cat:<18} {count:>4}  {bar}")
    return cats

def merge_all():
    print("=" * 55)
    print("Dataset Merger")
    print("=" * 55)

    sources = [
        os.path.join(DATA_DIR, 'raw', 'raw_cases_indiankanoon.json'),
        os.path.join(BASE, 'sample.json'),
        os.path.join(DATA_DIR, 'advanced_cases.json'),
    ]

    all_cases = []
    for src in sources:
        print(f"\nLoading: {os.path.basename(src)}")
        all_cases.extend(load_json_file(src))

    print(f"\nTotal before dedup: {len(all_cases)}")
    all_cases = deduplicate(all_cases)

    # Check class balance
    cats = check_class_balance(all_cases)

    # Save merged dataset
    out_json = os.path.join(DATA_DIR, 'processed', 'merged_cases.json')
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {out_json} ({len(all_cases)} records)")

    return all_cases

if __name__ == '__main__':
    merged = merge_all()
    print(f"\nFinal dataset: {len(merged)} cases ready for preprocessing.")
