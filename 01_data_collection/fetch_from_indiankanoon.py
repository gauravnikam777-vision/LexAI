"""
fetch_from_indiankanoon.py
==========================
Step 1 – Data Collection Script
Author: Gaurav Govind Nikam
Date: March 2025

Ye script Indian Kanoon website se legal cases fetch karne ki koshish karta hai.
Real API token nahi mila toh kuch data manually collect kiya aur kuch synthetic banaya.

Note: Indian Kanoon ka API limited hai — free users ke liye paginated results milte hain.
Humne jo data mila wo save kiya, baaki manually prepare kiya.
"""

import requests
import json
import time
import os

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
BASE_URL = "https://api.indiankanoon.org"
# NOTE: Ye token demo ke liye hai, real token ke liye register karein
# indiankanoon.org par account banana padega
API_TOKEN = "YOUR_INDIANKANOON_TOKEN_HERE"

# Categories jo humein chahiye thi
SEARCH_QUERIES = {
    "fraud":          ["cheque bounce IPC 138", "cyber fraud 420 IPC", "bank fraud"],
    "property":       ["property dispute encroachment", "builder fraud RERA", "illegal eviction"],
    "labour":         ["wrongful termination labour court", "unpaid wages employer", "POSH act complaint"],
    "family":         ["domestic violence 498A", "divorce maintenance 125 CrPC", "child custody dispute"],
    "consumer":       ["consumer complaint defective product", "insurance claim rejected", "e-commerce fraud"],
    "motor_accident": ["motor accident MACT claim", "hit and run compensation", "rash driving 304A"],
    "criminal":       ["bail application sessions court", "FIR registration refused", "anticipatory bail 438"],
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'raw_cases_indiankanoon.json')


def fetch_cases_for_query(query, category, max_pages=3):
    """
    Ek search query ke liye Indian Kanoon se cases fetch karo.
    Rate limiting ki wajah se sleep add kiya hai.
    """
    results = []
    headers = {"Authorization": f"Token {API_TOKEN}"}

    for page in range(0, max_pages):
        params = {
            "formInput": query,
            "pagenum": page,
        }
        try:
            response = requests.get(
                f"{BASE_URL}/search/",
                headers=headers,
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                docs = data.get("docs", [])

                if not docs:
                    print(f"  [!] No more results for '{query}' on page {page}")
                    break

                for doc in docs:
                    results.append({
                        "case_id":    doc.get("tid", ""),
                        "title":      doc.get("title", ""),
                        "text":       doc.get("headline", ""),   # summary text
                        "category":   category,
                        "source":     "indiankanoon",
                        "year":       doc.get("publishdate", "")[:4] if doc.get("publishdate") else "",
                    })

                print(f"  [OK] Page {page}: got {len(docs)} docs for '{query}'")
                time.sleep(1.5)  # rate limit — 1 request per 1.5 seconds

            elif response.status_code == 401:
                print(f"  [ERR] Auth failed — check API token")
                break
            elif response.status_code == 429:
                print(f"  [WAIT] Rate limited. Sleeping 10 seconds...")
                time.sleep(10)
            else:
                print(f"  [ERR] Status {response.status_code} for query '{query}'")
                break

        except requests.exceptions.ConnectionError:
            print(f"  [ERR] Connection failed. Running offline?")
            break

    return results


def collect_all_data():
    """
    Sab categories ke liye data collect karo aur JSON mein save karo.
    """
    all_cases = []
    total = 0

    print("=" * 60)
    print("Indian Kanoon Data Collector")
    print("=" * 60)

    for category, queries in SEARCH_QUERIES.items():
        print(f"\n[>>>] Category: {category.upper()}")
        for q in queries:
            print(f"  Fetching: '{q}'...")
            cases = fetch_cases_for_query(q, category)
            all_cases.extend(cases)
            total += len(cases)
            time.sleep(2)  # extra wait between queries

    print(f"\n[DONE] Total cases collected: {total}")
    print(f"[SAVE] Saving to {OUTPUT_FILE}...")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_cases, f, indent=2, ensure_ascii=False)

    print(f"[OK] Saved {len(all_cases)} cases.")
    return all_cases


def create_sample_manual_cases():
    """
    Jab API se data nahi mila, humne manually kuch cases likhe.
    Ye function sample cases return karta hai jo hum ne khud banaye.
    """
    # NOTE: Ye cases humne manually research karke banaye hain
    # based on actual Indian court judgments we read online
    manual_cases = [
        {
            "case_id": "MANUAL_001",
            "text": "My employer fired me without giving any notice or salary. I worked for 3 years and they suddenly terminated me saying company is closed. They also did not give PF money.",
            "category": "labour",
            "risk_level": "High",
            "outcome": "win",
            "source": "manual",
        },
        {
            "case_id": "MANUAL_002",
            "text": "Someone did UPI fraud with me. They called pretending to be bank officer and asked for OTP. I shared OTP and 45000 rupees got deducted from my account.",
            "category": "fraud",
            "risk_level": "High",
            "outcome": "win",
            "source": "manual",
        },
        {
            "case_id": "MANUAL_003",
            "text": "My landlord is forcibly trying to evict me without court order. He has cut electricity and water supply. I have paid all rent on time.",
            "category": "property",
            "risk_level": "Medium",
            "outcome": "win",
            "source": "manual",
        },
        # ... more cases were added manually
    ]
    return manual_cases


if __name__ == "__main__":
    print("Starting data collection...")
    print("Note: If API token is not set, only manual cases will be saved.\n")

    if API_TOKEN == "YOUR_INDIANKANOON_TOKEN_HERE":
        print("[WARN] API token not configured. Saving sample manual cases only.")
        cases = create_sample_manual_cases()
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved {len(cases)} manual cases to {OUTPUT_FILE}")
    else:
        collect_all_data()
