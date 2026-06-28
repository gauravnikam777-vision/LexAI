# Step 1 – Data Collection

## Objective
Indian legal cases ka ek achha dataset banana tha jisme real-world FIR types,
categories aur outcomes hon. Ye pehla aur sabse mushkil step tha.

## Sources We Tried

### 1. Indian Kanoon API (indiankanoon.org)
- Ye ek open legal database hai India ka
- Humne `fetch_from_indiankanoon.py` script likhi jo API se cases fetch karti hai
- Problem: Free tier mein rate limiting tha, bulk download allowed nahi tha
- Iske baad manually JSON mein convert kiya

### 2. Manual Labeling
- Kuch cases khud likhe aur label kiye (category + risk_level + outcome)
- Ya labeling 7 categories ke hisab se ki:
  - fraud, property, labour, family, consumer, motor_accident, criminal
- Risk label: Low / Medium / High (manually assigned based on case severity)

### 3. Synthetic Data Generation
- Real patterns ke basis par synthetic cases generate kiye
- `generate_law_dataset.py` script ke through

## Final Dataset

```
data/
├── raw/
│   └── raw_cases_indiankanoon.json   ← raw scraped data
├── processed/
│   ├── cases_cleaned.csv             ← after preprocessing
│   └── advanced_cases.csv            ← enhanced with IPC references
```

## Dataset Stats (Final)
- Total records: ~200+ cases
- Categories: 7 (balanced using manual labeling)
- Fields: case_id, text, category, risk_level, outcome, ipc_sections, what_to_do, how_to_do, when_to_do

## Lessons Learned
- Indian legal data is not easy to get — most portals are behind paywalls
- Had to mix real + synthetic data
- Data quality matters more than quantity for this use case
