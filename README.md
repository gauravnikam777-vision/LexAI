# LexAI – AI-Powered Indian Legal Assistance System
### Final Year Project | B.Tech Computer Science | Gaurav Govind Nikam

---

## 📌 Problem Statement

Legal help in India is expensive and inaccessible for most common people.
Most people don't know:
- Their rights when arrested
- How to file an FIR if police refuse
- What to do after a cheque bounce
- Which IPC section applies to their case

**LexAI** is an AI + ML based system that provides free, instant legal guidance
based on Indian law (IPC, BNS 2023, CrPC, BNSS 2023) in English and Hinglish.

---

## 🏗️ Project Architecture

```
[User] → [Flask Web App] → [ML Models] → [Legal Analysis]
                        ↓
                  [Gemini AI] (fallback for complex queries)
                        ↓
                  [SQLite Database]
```

---

## 🔄 ML Pipeline (Step by Step)

This project was built step by step, following a proper ML workflow:

### Step 1 – Data Collection (`01_data_collection/`)
- Tried Indian Kanoon API for real legal case data
- Also manually built dataset of ~200+ Indian legal cases
- 7 categories: fraud, property, labour, family, consumer, motor_accident, criminal
- Each case has: text, category, risk_level, outcome, IPC sections, action steps

### Step 2 – Preprocessing (`02_data_preprocessing/`)
- Text cleaning: lowercase, remove noise, preserve IPC section numbers
- Custom stopword list (NLTK stopwords removed legal keywords — had to make custom)
- Label encoding for category and risk_level
- TF-IDF vectorization (max_features=5000, bigrams)

### Step 3 – Model Training (`03_model_training/`)
- **Category Classifier**: TF-IDF + RandomForestClassifier → ~82% accuracy
- **Risk Classifier**: TF-IDF + RandomForestClassifier → ~75% accuracy  
- **Similarity Search**: Cosine similarity on TF-IDF matrix (RAG-style)
- Models saved as `.pkl` files in `models/`

### Step 4 – Integration
- Flask web app integrates ML models for real-time prediction
- Gemini AI as fallback for complex/unseen queries
- Rule-based intent matching for known questions (50+ patterns)
- Win probability derived from risk level (data-driven mapping)

### Step 5 – Web Application
- **4 user roles**: Admin, Lawyer, Client, Chatbot Manager
- Role-based dashboards with separate features for each role
- SQLite database for users, cases, lawyer notes, chatbot logs, appointments
- Chatbot Manager can add/remove custom intents without touching code

---

## 👥 User Roles

| Role | Can Register? | Key Features |
|------|--------------|-------------|
| **Client** | ✅ Yes | AI Chat, Submit Cases, Appointments |
| **Lawyer** | ✅ Yes | Case Queue, Review, Add Notes |
| **Admin** | ❌ Admin only | User Management, System Stats |
| **Chatbot Manager** | ❌ Admin only | Intent Management, Logs, Analytics |

**Demo accounts:**
```
client1 / client123
lawyer1 / lawyer123
admin1  / admin123
cbm1    / cbm123
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|----------|
| Frontend | HTML, CSS, Vanilla JS |
| Backend | Python + Flask |
| ML Models | scikit-learn (RandomForest, TF-IDF) |
| AI | Google Gemini API (free tier) |
| Database | SQLite |
| Fonts | Google Fonts (Inter) |

---

## 📁 Folder Structure

```
legal_ai/
├── 01_data_collection/      ← Dataset collection scripts
├── 02_data_preprocessing/   ← Text cleaning, encoding
├── 03_model_training/       ← Model training, evaluation
├── data/                    ← Dataset files (CSV, JSON)
├── models/                  ← Trained .pkl model files
├── templates/               ← HTML templates
├── app.py                   ← Main Flask application
├── legal_brain.py           ← AI response engine
└── requirements.txt
```

---

## ▶️ How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key in .env
# GEMINI_API_KEY=your_key_here

# 3. Run the app
python app.py

# 4. Open browser
# http://localhost:5000
```

---

## 📊 Model Performance

| Model | Accuracy |
|-------|---------|
| Category Classifier | ~82% |
| Risk Level Classifier | ~75% |

*(See `03_model_training/training_results.txt` for full evaluation)*

---

## 🔮 Future Work

- Fine-tune BERT/LegalBERT on larger Indian legal corpus
- Add Hindi language support (currently Hinglish works via Gemini)
- Integrate real-time Indian Kanoon API for live case lookup
- Add lawyer-client video consultation feature
- Deploy to cloud (Railway / Render) for public access

---

## 📝 Disclaimer

LexAI provides **informational guidance only** and does not constitute legal advice.
Always consult a qualified lawyer for your specific legal situation.

---

*Made with ❤️ by Gaurav Govind Nikam · MCA Final Year Project · 2026*
