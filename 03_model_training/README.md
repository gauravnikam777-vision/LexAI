# Step 3 – Model Training

## Models We Built

### 3.1 Category Classifier (`train_category_model.py`)
- **Algorithm**: RandomForestClassifier
- **Features**: TF-IDF vectors (5000 features, bigrams)
- **Target**: 7 categories (fraud, property, labour, family, consumer, motor_accident, criminal)
- **Best Accuracy**: ~82% on test set

### 3.2 Risk Level Classifier (`train_risk_model.py`)
- **Algorithm**: RandomForestClassifier
- **Features**: Same TF-IDF vectors
- **Target**: Low / Medium / High
- **Best Accuracy**: ~75% on test set

### 3.3 Similarity Search (`train_similarity_model.py`)
- **Technique**: TF-IDF Cosine Similarity (RAG-style retrieval)
- **Purpose**: Find similar past cases for context
- **Libraries**: sklearn cosine_similarity

## Evaluation (`model_evaluation.py`)
- Classification Report (Precision, Recall, F1)
- Confusion Matrix
- Category-wise accuracy breakdown

## Training Results
See `training_results.txt` for actual run output.

## How to Re-train

```bash
# First preprocess
python 02_data_preprocessing/clean_text.py

# Then train models
python 03_model_training/train_category_model.py
python 03_model_training/train_risk_model.py
python 03_model_training/train_similarity_model.py

# Evaluate
python 03_model_training/model_evaluation.py
```

## Notes
- Models are saved in `models/` as `.pkl` files
- Retrain when new data is added to `data/advanced_cases.json`
- Minimum ~50 cases per category for decent accuracy
