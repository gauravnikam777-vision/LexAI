# Step 2 – Data Preprocessing

## What is Preprocessing?

Raw data directly ML model mein nahi daal sakte.
Pehle clean karna padta hai — noise remove karo, text normalize karo, labels encode karo.

## Steps Followed

### 2.1 Text Cleaning (`clean_text.py`)
- Lowercase conversion
- Special characters / punctuation remove
- Extra whitespace trim
- Numbers partially preserved (IPC section numbers important hain)
- Hinglish words handle kiye (mixed Hindi-English)

### 2.2 Label Encoding
- `category` → LabelEncoder → integer
- `risk_level` → LabelEncoder → integer (Low=0, Medium=1, High=2)
- `outcome` → Mostly 'win', kept as binary

### 2.3 TF-IDF Vectorization
- `TfidfVectorizer` with `max_features=5000`
- `ngram_range=(1,2)` — unigrams aur bigrams dono
- `min_df=2` — rare words ignore
- Indian legal terms manually added to vocabulary

### 2.4 Train/Test Split
- 80% train, 20% test
- `stratify=y` use kiya taaki har class equally represented ho

## Problems Faced

1. **Imbalanced Data**: fraud aur criminal cases zyada the, family/consumer kam
   → Fix: Oversampled minority classes manually

2. **Short texts**: Kuch cases sirf 1-2 lines ki thi, model confuse ho raha tha
   → Fix: `min_df=2` threshold rakha, short texts filter kiye

3. **Hindi/Hinglish words**: NLTK ke English stopwords Indian legal words remove kar rahe the
   → Fix: Custom stopword list banai (dekho `preprocessing_notes.txt`)

## Output Files
- `data/processed/cases_cleaned.csv` — clean text with labels
- `models/vectorizer.pkl` — fitted TF-IDF vectorizer
- `models/category_encoder.pkl` — category label encoder
- `models/risk_encoder.pkl` — risk label encoder
