"""
app.py  –  LexAI – AI Legal Intelligence Platform
==================================================
Author  : Gaurav Govind Nikam
Project : MCA Final Year Project – AI/ML Law Assistant
Degree  : Master of Computer Applications (MCA) 2026
Date    : 2025–2026

Description:
    End-to-end Flask web app for Indian legal assistance.
    Uses ML models (TF-IDF + RandomForest) for case classification
    and rule-based + Gemini AI for chatbot responses.

Actors / Roles:
    admin           → Full platform control
    lawyer          → Case queue + review
    client          → Submit cases + AI chat
    chatbot_manager → Manage chatbot intents + monitor logs

ML Pipeline:
    See 01_data_collection/, 02_data_preprocessing/, 03_model_training/
"""

import os, pickle, sqlite3, hashlib, json, re
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from legal_brain import get_legal_response, normalize, INTENT_TO_CATEGORY

app = Flask(__name__)
# TODO: move this to .env before deploying
app.secret_key = 'lexai_secret_key_2025'

BASE    = os.path.dirname(__file__)
MODELS  = os.path.join(BASE, 'models')
DB_PATH = os.path.join(BASE, 'database.db')


# ──────────────────────────────────────────────────────────────────
#  HELPER: load pkl model files
# ──────────────────────────────────────────────────────────────────
def load(name):
    """Load a pkl file — warn on version mismatch but keep going."""
    path = os.path.join(MODELS, name)
    if not os.path.exists(path):
        print(f'[WARN] Model file missing: {name}')
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)

vectorizer   = load('vectorizer.pkl')
cat_model    = load('category_model.pkl')
cat_enc      = load('category_encoder.pkl')
risk_model   = load('risk_model.pkl')
risk_enc     = load('risk_encoder.pkl')
tfidf_matrix = load('tfidf_matrix.pkl')
case_records = load('case_records.pkl') or []

print("[OK] ML models loaded. Categories:", list(cat_enc.classes_) if cat_enc else '?')


# ══════════════════════════════════════════════════════════════════
#  DATABASE SETUP
# ══════════════════════════════════════════════════════════════════
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create all tables and seed default users.
    Called once on startup.
    """
    with get_db() as conn:
        conn.executescript("""
        -- Users table (4 roles: admin, lawyer, client, chatbot_manager)
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role     TEXT NOT NULL DEFAULT 'client',
            email    TEXT DEFAULT '',
            full_name TEXT DEFAULT '',
            created  TEXT DEFAULT (datetime('now'))
        );

        -- Cases submitted by clients
        CREATE TABLE IF NOT EXISTS cases (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            title        TEXT NOT NULL,
            description  TEXT NOT NULL,
            category     TEXT,
            risk_level   TEXT,
            outcome_pred TEXT,
            outcome_prob REAL,
            status       TEXT DEFAULT 'pending',
            created      TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        -- Lawyer notes on cases
        CREATE TABLE IF NOT EXISTS lawyer_notes (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id   INTEGER NOT NULL,
            lawyer_id INTEGER NOT NULL,
            note      TEXT NOT NULL,
            created   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (case_id)   REFERENCES cases(id),
            FOREIGN KEY (lawyer_id) REFERENCES users(id)
        );

        -- Chat messages inside case threads
        CREATE TABLE IF NOT EXISTS messages (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id INTEGER NOT NULL,
            sender  TEXT NOT NULL,
            role    TEXT NOT NULL,
            content TEXT NOT NULL,
            created TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (case_id) REFERENCES cases(id)
        );

        -- Appointment booking by clients
        CREATE TABLE IF NOT EXISTS appointments (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            client_id INTEGER NOT NULL,
            lawyer_id INTEGER,
            category  TEXT,
            note      TEXT DEFAULT '',
            status    TEXT DEFAULT 'requested',
            created   TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (client_id) REFERENCES users(id),
            FOREIGN KEY (lawyer_id) REFERENCES users(id)
        );

        -- Chatbot conversation logs (monitored by chatbot_manager)
        CREATE TABLE IF NOT EXISTS chatbot_logs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            username   TEXT DEFAULT 'anonymous',
            query      TEXT NOT NULL,
            response   TEXT NOT NULL,
            used_ai    INTEGER DEFAULT 0,   -- 1 = Gemini, 0 = rule-based/ML
            category   TEXT DEFAULT '',
            risk_level TEXT DEFAULT '',
            created    TEXT DEFAULT (datetime('now'))
        );

        -- Custom chatbot intents (managed by chatbot_manager)
        CREATE TABLE IF NOT EXISTS chatbot_intents (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            trigger  TEXT NOT NULL,         -- keyword(s) that trigger this
            response TEXT NOT NULL,         -- HTML response
            category TEXT DEFAULT 'general',
            active   INTEGER DEFAULT 1,
            created  TEXT DEFAULT (datetime('now'))
        );

        -- System config (admin can change settings)
        CREATE TABLE IF NOT EXISTS system_config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        -- Chat session history (server-side storage for full conversation memory)
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            history_json TEXT NOT NULL DEFAULT '[]',
            updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id)
        );
        """)

        # Seed default users if DB is empty
        cur = conn.execute("SELECT COUNT(*) FROM users")
        if cur.fetchone()[0] == 0:
            _hash = lambda p: hashlib.sha256(p.encode()).hexdigest()
            seed_users = [
                ('admin1',  _hash('admin123'),  'admin'),
                ('lawyer1', _hash('lawyer123'), 'lawyer'),
                ('client1', _hash('client123'), 'client'),
                ('cbm1',    _hash('cbm123'),    'chatbot_manager'),
            ]
            for uname, pwd, role in seed_users:
                conn.execute(
                    "INSERT INTO users(username,password,role) VALUES(?,?,?)",
                    (uname, pwd, role)
                )
            conn.commit()
            print("[OK] Default system users seeded:")
            print("     admin1/admin123 | lawyer1/lawyer123")
            print("     client1/client123 | cbm1/cbm123")

        # Seed default system config
        defaults = [
            ('chatbot_enabled', '1'),
            ('max_cases_per_client', '50'),
            ('platform_name', 'LexAI'),
        ]
        for key, val in defaults:
            conn.execute(
                "INSERT OR IGNORE INTO system_config(key,value) VALUES(?,?)",
                (key, val)
            )
        conn.commit()


init_db()


# ══════════════════════════════════════════════════════════════════
#  AUTH DECORATORS / HELPERS
# ══════════════════════════════════════════════════════════════════
def login_required(f):
    """Redirect to login if not authenticated."""
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return wrapper


def role_required(*roles):
    """Reject if user's role is not in the allowed roles list."""
    from functools import wraps
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if 'user_id' not in session:
                return jsonify({'error': 'Unauthorized'}), 401
            if session.get('role') not in roles:
                return jsonify({'error': 'Forbidden – insufficient permissions'}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator


# ══════════════════════════════════════════════════════════════════
#  WIN PROBABILITY (from risk level)
# ══════════════════════════════════════════════════════════════════
WIN_PROB_BY_RISK = {'Low': 85.0, 'Medium': 68.0, 'High': 52.0}


# ══════════════════════════════════════════════════════════════════
#  LAW MAP  (applicable statutes per category)
# ══════════════════════════════════════════════════════════════════
LAW_MAP = {
    'property': [
        'Transfer of Property Act 1882',
        'Registration Act 1908 – mandatory for immovable property >100',
        'Specific Relief Act 1963 – for injunctions and possession',
        'RERA Act 2016 – for builder/developer disputes',
        'Indian Easements Act 1882',
        'Section 441 IPC / Section 329 BNS – Criminal Trespass',
        'State Rent Control Act (varies by state)',
    ],
    'labour': [
        'Industrial Disputes Act 1947',
        'Payment of Wages Act 1936',
        'Minimum Wages Act 1948',
        'EPF and Miscellaneous Provisions Act 1952',
        'Maternity Benefit Act 1961',
        'POSH Act 2013 (Sexual Harassment at Workplace)',
        'Factories Act 1948',
        'Payment of Gratuity Act 1972',
        'State Shops and Establishments Act',
    ],
    'fraud': [
        'IPC Section 420 (Cheating) / BNS Section 318',
        'IPC Section 406 (Criminal Breach of Trust)',
        'IPC Section 467/468 (Forgery)',
        'IT Act 2000 Sections 66C, 66D (Identity/Cyber Fraud)',
        'Section 138, Negotiable Instruments Act 1881 (Cheque Bounce)',
        'SEBI Act 1992 (Securities Fraud)',
        'RERA Act 2016 (Builder Fraud)',
        'Consumer Protection Act 2019',
        'Prevention of Money Laundering Act 2002',
    ],
    'family': [
        'Hindu Marriage Act 1955',
        'Special Marriage Act 1954 (Interfaith / Court Marriage)',
        'Hindu Succession Act 1956',
        'Domestic Violence Act 2005',
        'Section 498A IPC (Dowry Cruelty)',
        'Dowry Prohibition Act 1961',
        'Section 125 CrPC / Section 144 BNSS (Maintenance)',
        'Maintenance and Welfare of Parents & Senior Citizens Act 2007',
        'Juvenile Justice Act 2015 (Adoption)',
    ],
    'consumer': [
        'Consumer Protection Act 2019',
        'IRDAI Regulations (Insurance Disputes)',
        'RERA Act 2016 (Real Estate Consumers)',
        'Electricity Act 2003',
        'TRAI Regulations (Telecom Disputes)',
        'Indian Contract Act 1872 (Misrepresentation)',
        'IT Act 2000 (Online Fraud)',
    ],
    'motor_accident': [
        'Motor Vehicles Act 1988',
        'IPC Section 279 (Rash Driving)',
        'IPC Section 304A (Causing Death by Negligence)',
        'IPC Section 337/338 (Hurt by Negligent Act)',
        'Insurance Act 1938',
        'Solatium Fund Scheme (for unidentified vehicles)',
        'Motor Accident Claims Tribunal (MACT) – Chapter XII MV Act',
    ],
    'criminal': [
        'IPC / Bharatiya Nyaya Sanhita (BNS) 2023',
        'CrPC / Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023',
        'Article 22 Constitution of India (Rights on Arrest)',
        'Section 154 CrPC / Section 173 BNSS (FIR Registration)',
        'Section 436-439 CrPC (Bail Provisions)',
        'Section 438 CrPC (Anticipatory Bail)',
        'Section 499/500 IPC (Defamation)',
        'Section 383-389 IPC (Extortion)',
        'Domestic Violence Act 2005',
    ],
    'sexual_offense': [
        'IPC Section 375/376 / BNS Section 63 (Rape)',
        'IPC Section 354 / BNS Section 73 (Outraging Modesty)',
        'POCSO Act 2012',
        'Criminal Law (Amendment) Act 2013/2018',
        'CrPC Section 164(5A) (Statement Recording)',
    ],
    'theft_robbery': [
        'IPC Section 378/379 / BNS Section 303 (Theft)',
        'IPC Section 390/392 / BNS Section 309 (Robbery)',
        'IPC Section 391/395 / BNS Section 310 (Dacoity)',
        'IPC Section 411 / BNS Section 317 (Receiving Stolen Property)',
    ],
    'cyber_crime': [
        'IT Act 2000 Section 66 (Computer Related Offences)',
        'IT Act Section 66C (Identity Theft)',
        'IT Act Section 66D (Cheating by Personation)',
        'IT Act Section 67 (Publishing Obscene Material)',
        'IPC Section 420 / BNS Section 318 (Cheating)',
    ],
    'medical_negligence': [
        'Consumer Protection Act 2019 (Deficiency of Service)',
        'IPC Section 304A / BNS Section 106(1) (Death by Negligence)',
        'Indian Medical Council (Professional Conduct, Etiquette and Ethics) Regulations 2002',
        'Clinical Establishments (Registration and Regulation) Act 2010',
    ],
    'constitutional': [
        'Constitution of India Article 32 (Supreme Court Writ Jurisdiction)',
        'Constitution of India Article 226 (High Court Writ Jurisdiction)',
        'Constitution of India Article 14 (Equality Before Law)',
        'Constitution of India Article 21 (Right to Life and Personal Liberty)',
    ],
    'corporate_tax': [
        'Income Tax Act 1961',
        'Central Goods and Services Tax (CGST) Act 2017',
        'Companies Act 2013',
        'Insolvency and Bankruptcy Code (IBC) 2016',
        'FEMA 1999',
    ],
}

# ══════════════════════════════════════════════════════════════════
#  STRATEGY TEMPLATES
# ══════════════════════════════════════════════════════════════════
STRATEGY = {
    'property': {
        'High': [
            'IMMEDIATELY secure all original documents: sale deed, khata, mutation record, title deed.',
            'Apply for an urgent Stay Order / Temporary Injunction in Civil Court.',
            'File an FIR under Section 441 IPC (Criminal Trespass) if physical possession is at stake.',
            'Engage a licensed government surveyor to officially measure and record boundaries.',
            'If builder fraud, file a RERA complaint at your State RERA Authority portal.',
            'Consult a property lawyer within 48 hours for a declaration of title suit.',
        ],
        'Medium': [
            'Gather all property documents: sale deed, tax receipts, rent agreement, electricity bills.',
            'Send a formal Legal Notice to the opposing party giving 30 days to resolve the matter.',
            'If not resolved, file a civil suit in the District Civil Court.',
            'If landlord-tenant dispute, review your State Rent Control Act for specific protections.',
        ],
        'Low': [
            'Send a formal written legal notice specifying your claim and giving 15 days to respond.',
            'Attempt mediation or arbitration before approaching court.',
            'Gather all documents: rent receipts, title proof, agreement copies.',
        ],
    },
    'labour': {
        'High': [
            'File a complaint immediately with the District Labour Commissioner.',
            'Preserve all evidence: appointment letter, salary slips, attendance records.',
            'Send a formal Legal Notice to employer via Registered Post within 15 days.',
            'File an FIR if criminal intimidation or PF fraud is involved.',
            'For sexual harassment: file with Internal Complaints Committee (ICC) within 3 months.',
        ],
        'Medium': [
            'Send a registered Legal Notice to employer specifying violation.',
            'File complaint with the Labour Commissioner.',
            'Collect all evidence: appointment letter, salary slips, bank statements.',
        ],
        'Low': [
            'Send a formal demand letter to employer via registered post.',
            'Approach HR department with a written complaint first.',
            'File with Labour Commissioner if no response within 30 days.',
        ],
    },
    'fraud': {
        'High': [
            'IMMEDIATELY call Cyber Crime Helpline 1930 and file at cybercrime.gov.in.',
            'Preserve ALL evidence: screenshots, email chains, transaction IDs, SMS.',
            'Contact your bank to freeze / reverse fraudulent transactions immediately.',
            'File FIR at nearest police station or Cyber Crime Cell.',
            'For cheque bounce: send Legal Notice within 30 days of bounce memo.',
        ],
        'Medium': [
            'File police complaint and Cyber Crime Cell complaint with all evidence.',
            'Send a Legal Demand Notice to the fraudster.',
            'Contact bank to block further unauthorized transactions.',
        ],
        'Low': [
            'Send a formal legal notice demanding return of money or goods.',
            'File a complaint with Consumer Forum (CONFONET) for consumer disputes.',
            'File complaint with local police station with all supporting documents.',
        ],
    },
    'family': {
        'High': [
            'Call Women Helpline 181 or Police 112 immediately if you are in danger.',
            'File complaint under Domestic Violence Act 2005 with Protection Officer.',
            'File FIR under Section 498A IPC (dowry cruelty) at nearest police station.',
            'Apply for interim maintenance under Section 125 CrPC in Magistrate Court.',
        ],
        'Medium': [
            'Consult a family lawyer to understand your rights under applicable personal law.',
            'Send a formal legal notice to the opposing party.',
            'Apply for interim maintenance if you lack financial support.',
            'File a petition in Family Court for the relevant relief.',
        ],
        'Low': [
            'Consult a family lawyer to understand mutual consent divorce process.',
            'Both parties agree on: alimony, child custody, property division — document in writing.',
            'File joint petition in Family Court after 1 year of separation.',
        ],
    },
    'consumer': {
        'High': [
            'File grievance with IRDAI Bima Bharosa portal for insurance issues.',
            'Send a Legal Notice to company giving 15 days to replace/refund.',
            'File consumer complaint at edaakhil.nic.in (District Consumer Commission).',
        ],
        'Medium': [
            'Send formal complaint to company Grievance Redressal Officer first.',
            'Call National Consumer Helpline: 1800-11-4000.',
            'File e-Daakhil complaint at District Consumer Commission.',
        ],
        'Low': [
            'Send written complaint to company giving 15 days to resolve.',
            'Register grievance on National Consumer Helpline (NCH): 1800-11-4000.',
            'File e-Daakhil online at District Consumer Commission if unresolved.',
        ],
    },
    'motor_accident': {
        'High': [
            'Call 112 immediately. Do NOT leave the accident scene.',
            'Note the vehicle number, color, make, and get witness contact details.',
            'Get a Medico-Legal Certificate (MLC) from a government hospital.',
            'File FIR at nearest police station under IPC Sections 279 and 337/338.',
            'File claim petition in Motor Accident Claims Tribunal (MACT) within 6 months.',
        ],
        'Medium': [
            'File FIR at nearest police station with all accident details.',
            'Get vehicle damage assessed in writing by authorized garage.',
            'Collect CCTV evidence within 7 days.',
            'File MACT claim with medical bills, FIR copy, and RC/insurance documents.',
        ],
        'Low': [
            'Inform your insurance company about the accident within 7 days.',
            'Get a written damage estimate from an authorized garage.',
            'File police complaint for the record.',
        ],
    },
    'criminal': {
        'High': [
            'You have the RIGHT to know the reason for arrest under Article 22 of the Constitution.',
            'Demand to contact a lawyer and inform a family member immediately upon arrest.',
            'Police MUST produce you before a Magistrate within 24 hours of arrest.',
            'Apply for bail — bailable offenses at police station; non-bailable at Magistrate/Sessions Court.',
        ],
        'Medium': [
            'File FIR at nearest police station immediately with full details.',
            'Preserve all evidence: screenshots, recordings, witnesses.',
            'Send legal notice to the accused for civil remedies.',
        ],
        'Low': [
            'File a written police complaint with all relevant evidence.',
            'If police refuse, escalate to SP or Magistrate Court.',
            'Consult a criminal lawyer to evaluate the best course of action.',
        ],
    },
    'sexual_offense': {
        'High': [
            'Call 112 or Women Helpline 1091 / 181 immediately.',
            'Go to the nearest government hospital for medical examination without washing or changing clothes.',
            'File a Zero FIR at any police station; they cannot refuse.',
            'Consult a specialized lawyer for rape and sexual assault cases immediately.',
        ],
        'Medium': [
            'Document all evidence: chats, emails, photos, witness accounts.',
            'File an FIR at the local police station.',
            'Seek legal and psychological counseling.',
        ],
        'Low': [
            'Consult a lawyer to explore your legal options safely.',
            'File a complaint with the police or Women\'s Cell.',
        ],
    },
    'theft_robbery': {
        'High': [
            'Call 112 immediately to report the robbery.',
            'Do not touch anything at the crime scene; preserve for forensics.',
            'File an FIR immediately and get a copy.',
            'Block all stolen credit/debit cards and phones.',
        ],
        'Medium': [
            'File an FIR detailing all stolen items with their receipts/IMEI numbers.',
            'Notify your insurance company if the stolen items were insured.',
        ],
        'Low': [
            'Register an e-FIR if your state allows for minor thefts.',
            'Keep an inventory of stolen items for records.',
        ],
    },
    'cyber_crime': {
        'High': [
            'Call Cyber Crime Helpline 1930 immediately.',
            'Register a complaint on cybercrime.gov.in.',
            'Contact your bank to freeze transactions/accounts.',
            'Do not delete any chats, links, emails, or call logs; take screenshots.',
        ],
        'Medium': [
            'File a complaint on the National Cyber Crime Reporting Portal.',
            'Report the fraudulent profile/website to the respective platform.',
        ],
        'Low': [
            'Report the incident to the platform where it occurred.',
            'Update passwords and enable Two-Factor Authentication (2FA).',
        ],
    },
    'medical_negligence': {
        'High': [
            'Secure all medical records, test reports, prescriptions, and bills immediately.',
            'Get an independent medical opinion detailing the negligence.',
            'File a complaint with the State Medical Council.',
            'Consult a specialized medical negligence lawyer to file a Consumer Court claim.',
        ],
        'Medium': [
            'Request complete medical records from the hospital in writing.',
            'Send a Legal Notice to the doctor/hospital.',
            'Prepare to file a case in the District Consumer Commission.',
        ],
        'Low': [
            'Seek a second medical opinion.',
            'Send a formal grievance letter to the hospital administration.',
        ],
    },
    'constitutional': {
        'High': [
            'If illegally detained, have a lawyer file a Habeas Corpus writ petition immediately.',
            'If fundamental rights are violated, prepare to file a Writ Petition under Article 32/226.',
            'Consult a senior constitutional lawyer without delay.',
        ],
        'Medium': [
            'Gather all government orders/notifications that violate your rights.',
            'File an RTI to get official reasons/documents.',
            'Consult a High Court lawyer to explore a writ petition.',
        ],
        'Low': [
            'File representations to the concerned government department.',
            'Use RTI Act to gather information before moving court.',
        ],
    },
    'corporate_tax': {
        'High': [
            'If facing a raid/survey, contact a senior tax lawyer immediately.',
            'Do not destroy any financial records; cooperate with officials.',
            'Reply to show-cause notices strictly within the deadline with legal assistance.',
        ],
        'Medium': [
            'Consult a Chartered Accountant and Tax Lawyer to draft a response to notices.',
            'File an appeal before the CIT(Appeals) within 30 days of an adverse order.',
        ],
        'Low': [
            'Rectify any errors in filed returns before the deadline.',
            'Seek professional advice on tax planning and compliance.',
        ],
    },
}

RISK_ADVICE = {
    'High':   '⚠️ URGENT: High-risk case. Engage a specialist lawyer within 48 hours.',
    'Medium': '⏳ MODERATE RISK: Document all evidence carefully and send a formal legal notice.',
    'Low':    '✅ LOW RISK: Consider alternative dispute resolution (mediation) first.',
}


def generate_strategy(category, risk_level):
    cat_s = STRATEGY.get(category, STRATEGY.get('fraud', {}))
    steps = cat_s.get(risk_level, cat_s.get('High', ['Consult a qualified legal professional.']))
    return {
        'steps':           steps,
        'advice':          RISK_ADVICE.get(risk_level, ''),
        'applicable_laws': LAW_MAP.get(category, ['Consult the relevant statutes with a qualified lawyer.']),
    }


# ══════════════════════════════════════════════════════════════════
#  ML ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════════════
def ml_analyze(text):
    """
    Run text through our trained ML models:
    - Category classification (TF-IDF + RandomForest)
    - Risk level classification (TF-IDF + RandomForest)
    - Similar case retrieval (cosine similarity RAG)

    NOTE: Uses try/except for predict_proba because models were trained
    on sklearn 1.8.0 but running on 1.5.1 — predict() works fine though.
    """
    if vectorizer is None or cat_model is None:
        # ML models not loaded — return a default result
        return {
            'category': 'criminal', 'cat_confidence': 0,
            'risk_level': 'Medium', 'risk_confidence': 0,
            'outcome': 'win', 'win_probability': 60.0,
            'similar_cases': [], 'advanced_strategy': None,
        }

    vec = vectorizer.transform([text])

    # ── Category prediction ──
    cat_idx  = cat_model.predict(vec)[0]
    category = cat_enc.inverse_transform([cat_idx])[0]
    # predict_proba may fail if sklearn version mismatch — graceful fallback
    try:
        cat_proba = cat_model.predict_proba(vec)[0]
        cat_conf  = round(float(cat_proba.max()) * 100, 1)
    except Exception:
        # Can't get probabilities — use decision function or fixed value
        try:
            scores   = cat_model.decision_function(vec)[0]
            scores   = np.exp(scores) / np.exp(scores).sum()   # softmax
            cat_conf = round(float(scores.max()) * 100, 1)
        except Exception:
            cat_conf = 75.0   # reasonable default if all else fails

    # ── Risk prediction ──
    risk_idx = risk_model.predict(vec)[0]
    risk_lvl = risk_enc.inverse_transform([risk_idx])[0]
    try:
        risk_proba = risk_model.predict_proba(vec)[0]
        risk_conf  = round(float(risk_proba.max()) * 100, 1)
    except Exception:
        try:
            scores    = risk_model.decision_function(vec)[0]
            scores    = np.exp(scores) / np.exp(scores).sum()
            risk_conf = round(float(scores.max()) * 100, 1)
        except Exception:
            risk_conf = 70.0

    # Win probability derived from predicted risk level
    win_prob = WIN_PROB_BY_RISK.get(risk_lvl, 60.0)

    # ── RAG: cosine similarity for top-3 similar past cases ──
    similar, best_rag = [], None
    if tfidf_matrix is not None and len(case_records) > 0:
        sims = cosine_similarity(vec, tfidf_matrix).flatten()
        top4 = sims.argsort()[-4:][::-1]
        for idx in top4:
            if idx < len(case_records) and sims[idx] > 0.04:
                rec = case_records[idx]
                if best_rag is None:
                    best_rag = rec
                similar.append({
                    'case_id'   : rec.get('case_id', ''),
                    'text'      : rec.get('text', '')[:130] + '…',
                    'category'  : rec.get('category', ''),
                    'risk_level': rec.get('risk_level', ''),
                    'outcome'   : rec.get('outcome', 'win'),
                    'similarity': round(float(sims[idx]) * 100, 1),
                })
            if len(similar) == 3:
                break

    # ── Advanced strategy from best matching dataset case ──
    advanced_strategy = None
    if best_rag:
        try:
            raw    = best_rag.get('how_to_do', '[]')
            how_to = json.loads(raw) if isinstance(raw, str) else raw
            if not isinstance(how_to, list):
                how_to = []
        except Exception:
            how_to = []
        advanced_strategy = {
            'ipc_sections': best_rag.get('ipc_sections', ''),
            'what_to_do'  : best_rag.get('what_to_do', ''),
            'how_to_do'   : how_to,
            'when_to_do'  : best_rag.get('when_to_do', ''),
        }

    return {
        'category'         : category,
        'cat_confidence'   : cat_conf,
        'risk_level'       : risk_lvl,
        'risk_confidence'  : risk_conf,
        'outcome'          : 'win',
        'win_probability'  : win_prob,
        'similar_cases'    : similar,
        'advanced_strategy': advanced_strategy,
    }


# ══════════════════════════════════════════════════════════════════
#  RULE-BASED LEGAL INTENT ENGINE  (50+ patterns)
# ══════════════════════════════════════════════════════════════════
LEGAL_INTENTS = [
    {
        'id': 'capabilities',
        'keywords': ['what can you do','capabilities','who are you','what are you','help me','what do you know','who created you','how can you help'],
        'answer': (
            "<strong>⚖️ LexAI – AI Legal Assistant</strong><br><br>"
            "I am trained on Indian Law and can provide accurate guidance on:<br>"
            "• 🏠 <strong>Property</strong>: Disputes, encroachment, builder fraud, RERA<br>"
            "• 👷 <strong>Labour</strong>: Wrongful termination, unpaid wages, POSH, PF fraud<br>"
            "• 💸 <strong>Fraud</strong>: Cyber crime, cheque bounce, OTP scam, Ponzi schemes<br>"
            "• 👨‍👩 <strong>Family</strong>: Divorce, maintenance, DV, dowry harassment, inheritance<br>"
            "• 🛒 <strong>Consumer</strong>: Defective product, insurance rejection, RERA, e-commerce<br>"
            "• 🚗 <strong>Motor Accident</strong>: Hit-and-run, MACT claim, insurance disputes<br>"
            "• 🚓 <strong>Criminal</strong>: FIR, bail, extortion, defamation, arrest rights<br><br>"
            "Simply describe your legal situation and I'll analyze it!"
        )
    },
    {
        'id': 'hit_and_run',
        'keywords': ['hit and run','hit someone','ran by car','run by car','car accident','vehicle accident','road accident','hit by car','bike accident','motor accident','knocked down'],
        'answer': (
            "🚗 <strong>Motor Vehicle Accident / Hit and Run (Indian Law)</strong><br><br>"
            "<strong>If you are the VICTIM:</strong><br>"
            "1. Call <strong>112</strong> immediately. Note vehicle number, color, location.<br>"
            "2. Get a <strong>Medico-Legal Certificate (MLC)</strong> from a government hospital.<br>"
            "3. File an <strong>FIR</strong> under IPC Sections 279 (Rash Driving) and 337/338.<br>"
            "4. File a compensation claim in <strong>MACT (Motor Accident Claims Tribunal)</strong>.<br>"
            "5. If driver fled and is not found, claim from the <strong>Solatium Fund</strong>.<br><br>"
            "⚖️ <strong>Laws:</strong> Motor Vehicles Act 1988, IPC Sections 279, 304A, 337."
        )
    },
    {
        'id': 'cheque_bounce',
        'keywords': ['cheque bounce','cheque dishonour','bounced cheque','check bounce','ni act','cheque returned','check returned','dishonoured cheque'],
        'answer': (
            "🏦 <strong>Cheque Bounce – Section 138 NI Act</strong><br><br>"
            "1. <strong>Send legal demand notice</strong> to issuer within <strong>30 days</strong> of the bounce memo via Registered Post.<br>"
            "2. The issuer has <strong>15 days</strong> to pay after receiving your notice.<br>"
            "3. If unpaid, file a <strong>criminal complaint in Magistrate Court</strong> within 30 days of the 15-day expiry.<br>"
            "4. You can claim the <strong>cheque amount + up to 2× as compensation</strong>.<br><br>"
            "<span style='color:#f08080'>⚠️ STRICT DEADLINE: Missing the 30-day window permanently bars your case.</span><br><br>"
            "⚖️ <strong>Law:</strong> Section 138, Negotiable Instruments Act 1881."
        )
    },
    {
        'id': 'domestic_violence',
        'keywords': ['domestic violence','wife beating','husband beating','physical abuse','domestic abuse','dowry harassment','in-laws harassing','spouse abuse'],
        'answer': (
            "👨‍👩‍👧 <strong>Domestic Violence / Dowry Harassment</strong><br><br>"
            "1. 🚨 Call <strong>Women's Helpline 181</strong> or <strong>Police 112</strong> immediately if in danger.<br>"
            "2. File a Domestic Incident Report with the nearest <strong>Protection Officer</strong>.<br>"
            "3. File complaint under the <strong>Domestic Violence Act 2005</strong> — Protection Order possible same day.<br>"
            "4. For dowry harassment, file FIR under <strong>IPC Section 498A</strong> (up to 3 years imprisonment).<br>"
            "5. Apply for <strong>interim maintenance</strong> under Section 125 CrPC.<br><br>"
            "⚖️ <strong>Laws:</strong> DV Act 2005, IPC Section 498A, Dowry Prohibition Act 1961."
        )
    },
    {
        'id': 'cyber_crime',
        'keywords': ['cyber crime','online fraud','hacked','phishing','upi fraud','bank fraud','otp fraud','sextortion','social media hacked','scammed online','net banking fraud'],
        'answer': (
            "💻 <strong>Cyber Crime / Online Fraud</strong><br><br>"
            "1. 🚨 <strong>IMMEDIATELY call Cyber Crime Helpline: 1930</strong>.<br>"
            "2. File at <strong>cybercrime.gov.in</strong> — National Cyber Crime Reporting Portal.<br>"
            "3. File FIR at your local police Cyber Cell.<br>"
            "4. Block all cards and accounts with your bank instantly.<br><br>"
            "<span style='color:#f08080'>⏰ Act within 72 hours — significantly increases money recovery chances.</span><br><br>"
            "⚖️ <strong>Laws:</strong> IT Act 2000 Sections 66C, 66D; IPC Section 420."
        )
    },
    {
        'id': 'divorce',
        'keywords': ['divorce','separation','marriage dissolution','mutual consent divorce','contested divorce','alimony','maintenance','want to divorce'],
        'answer': (
            "👨‍👩 <strong>Divorce / Separation (Indian Law)</strong><br><br>"
            "<strong>Mutual Consent Divorce (Fastest Route):</strong><br>"
            "1. Both parties file joint petition in Family Court.<br>"
            "2. 6-month cooling-off period (waivable by Supreme Court if no children).<br>"
            "3. Attend second motion hearing for final divorce decree.<br>"
            "<em>Requirement:</em> Must have lived separately for at least <strong>1 year</strong>.<br><br>"
            "<strong>Contested Divorce:</strong> File citing valid grounds in Family Court. Duration: 1–5+ years.<br><br>"
            "⚖️ <strong>Laws:</strong> Hindu Marriage Act 1955, Special Marriage Act 1954, Sec 125 CrPC."
        )
    },
    {
        'id': 'consumer_complaint',
        'keywords': ['consumer complaint','defective product','service deficiency','refund not received','online shopping fraud','amazon','flipkart','warranty','product defective','company cheated'],
        'answer': (
            "🛒 <strong>Consumer Complaint (Consumer Protection Act 2019)</strong><br><br>"
            "1. Send <strong>written complaint to company</strong> first — 15 days to resolve.<br>"
            "2. Call <strong>National Consumer Helpline: 1800-11-4000</strong>.<br>"
            "3. File at <strong>District Consumer Commission (DCDRC)</strong> for claims up to ₹50 Lakhs (free online: edaakhil.nic.in).<br><br>"
            "⏰ Time limit: <strong>2 years</strong> from the date of the defect.<br><br>"
            "⚖️ <strong>Law:</strong> Consumer Protection Act 2019."
        )
    },
    {
        'id': 'bail_arrest',
        'keywords': ['bail','arrested','police arrested','in custody','anticipatory bail','get bail','wrongful arrest','detained','custody'],
        'answer': (
            "⚖️ <strong>Bail / Arrest Rights (Indian Law)</strong><br><br>"
            "1. You have the RIGHT to know the <strong>reason for arrest</strong> (Article 22, Constitution of India).<br>"
            "2. Must be produced before a <strong>Magistrate within 24 hours</strong>.<br>"
            "3. <strong>Bailable offenses</strong>: Get bail at the police station itself.<br>"
            "4. <strong>Non-bailable offenses</strong>: Apply for bail in Magistrate or Sessions Court.<br>"
            "5. <strong>Anticipatory Bail</strong>: File under Section 438 CrPC in Sessions Court if you <em>fear</em> arrest.<br><br>"
            "⚖️ <strong>Laws:</strong> Sections 436-439 CrPC / BNSS; Article 22 of the Constitution."
        )
    },
    {
        'id': 'fir',
        'keywords': ['how to file fir','file fir','police not taking complaint','police not registering','zero fir','police refusing fir','police refusing to register'],
        'answer': (
            "📋 <strong>How to File an FIR (Indian Law)</strong><br><br>"
            "1. Go to the nearest police station and narrate the incident clearly.<br>"
            "2. Police <strong>MUST</strong> register FIR for cognizable offenses — it is your legal right under Section 154 CrPC.<br>"
            "3. If police <strong>refuse</strong>:<br>"
            "&nbsp;&nbsp;• Send written complaint by Registered Post to the <strong>Superintendent of Police (SP)</strong>.<br>"
            "&nbsp;&nbsp;• File a <strong>private complaint</strong> directly in the Magistrate's Court under Section 156(3) CrPC.<br>"
            "4. <strong>Zero FIR</strong>: You can file an FIR at <em>any</em> police station regardless of jurisdiction.<br><br>"
            "⚖️ <strong>Law:</strong> Section 154 CrPC / Section 173 BNSS 2023."
        )
    },
    {
        'id': 'labour_dispute',
        'keywords': ['job terminated','fired from job','wrongful termination','unpaid salary','salary not paid','employer not paying','unfair dismissal','retrenchment without pay'],
        'answer': (
            "👷 <strong>Labour / Employment Dispute</strong><br><br>"
            "1. Send a <strong>formal legal notice</strong> to employer via Registered Post — 15 days to comply.<br>"
            "2. File complaint with the <strong>District Labour Commissioner</strong>.<br>"
            "3. For wrongful termination: file case in <strong>Industrial Tribunal / Labour Court</strong>.<br>"
            "4. For unpaid wages: file under <strong>Payment of Wages Act 1936</strong>.<br>"
            "5. For PF fraud: file on <strong>EPFiGMS portal (epfigms.gov.in)</strong>.<br><br>"
            "⚖️ <strong>Laws:</strong> Industrial Disputes Act 1947, Payment of Wages Act 1936, EPF Act 1952."
        )
    },
    {
        'id': 'rera',
        'keywords': ['rera','builder fraud','flat possession delayed','builder not giving possession','real estate fraud','developer fraud','construction not started'],
        'answer': (
            "🏗️ <strong>Builder / RERA Complaint</strong><br><br>"
            "1. Collect: builder-buyer agreement, all payment receipts, brochures.<br>"
            "2. File <strong>RERA complaint online</strong> with your State's RERA Authority — demand possession or refund with <strong>10.85% interest</strong>.<br>"
            "3. If no RERA registration: approach <strong>NCDRC (National Consumer Commission)</strong>.<br>"
            "4. File a <strong>parallel FIR</strong> for cheating under Section 420 IPC.<br><br>"
            "⚖️ <strong>Laws:</strong> RERA Act 2016, Consumer Protection Act 2019, Section 420 IPC."
        )
    },
    {
        'id': 'extortion',
        'keywords': ['extortion','blackmail','threatening money','goons demanding money','extorted','protection money','being blackmailed'],
        'answer': (
            "🚨 <strong>Extortion / Blackmail</strong><br><br>"
            "1. <strong>Do NOT pay</strong> any extortion amount — it encourages more demands.<br>"
            "2. Preserve ALL threats: voice recordings, messages, CCTV footage.<br>"
            "3. File FIR at local police station for <strong>extortion under Section 383 IPC</strong>.<br>"
            "4. If online blackmail (sextortion): additionally file at <strong>cybercrime.gov.in</strong>.<br><br>"
            "⚖️ <strong>Laws:</strong> Sections 383-389 IPC / Section 308 BNS (Extortion)."
        )
    },
]

TYPO_MAP = {
    'landlaurd':'landlord','landlored':'landlord','roomate':'roommate',
    'divorse':'divorce','divorece':'divorce','divorc':'divorce',
    'cheq':'cheque','chq':'cheque','lazyer':'lawyer','lawer':'lawyer',
    'propety':'property','proprty':'property','labur':'labour','labourer':'labour',
}


def check_legal_intent(query):
    """Score all keywords against query; return best-matching intent answer, or None."""
    words = query.split()
    corrected = ' '.join(TYPO_MAP.get(w, w) for w in words)
    q = corrected.strip()

    # Also check DB for custom intents added by chatbot_manager
    try:
        with get_db() as conn:
            custom = conn.execute(
                "SELECT trigger, response FROM chatbot_intents WHERE active=1"
            ).fetchall()
        for row in custom:
            if row['trigger'].lower() in q:
                return row['response']
    except Exception:
        pass  # DB might not be ready yet

    best_score, best_answer = 0, None
    for intent in LEGAL_INTENTS:
        for kw in intent['keywords']:
            if kw in q:
                score = len(kw)
                if score > best_score:
                    best_score, best_answer = score, intent['answer']
                break
            kw_words = set(kw.split())
            if len(kw_words) > 1:
                matched = sum(1 for w in kw_words if w in q)
                if matched >= max(2, len(kw_words) - 1):
                    score = matched * 2
                    if score > best_score:
                        best_score, best_answer = score, intent['answer']

    return best_answer


def generate_document(case_title, category, risk_level, outcome, description):
    date = datetime.now().strftime('%d %B %Y')
    return f"""LEGAL NOTICE
Date: {date}

Subject: Notice regarding {category.title()} Law matter – {case_title}

To Whom It May Concern,

I hereby bring to your attention the following matter:

CASE SUMMARY:
{description}

CATEGORY: {category.replace('_', ' ').title()} Law
ASSESSED RISK: {risk_level}
PREDICTED OUTCOME: {outcome.title()}

You are hereby notified to resolve this matter within 30 days of receipt of
this notice, failing which appropriate legal proceedings will be initiated without
further notice.

⚠️ DISCLAIMER: This document is AI-generated for informational purposes only
and does not constitute formal legal advice. Please consult a qualified lawyer
before taking any legal action.

Regards,
[Your Name]
[Contact Information]
[Date: {date}]
"""


# ══════════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════════
@app.route('/')
def index():
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data  = request.get_json()
        uname = data.get('username', '').strip()
        pwd   = hashlib.sha256(data.get('password', '').encode()).hexdigest()
        with get_db() as conn:
            user = conn.execute(
                "SELECT * FROM users WHERE username=? AND password=?", (uname, pwd)
            ).fetchone()
        if user:
            session['user_id']  = user['id']
            session['username'] = user['username']
            session['role']     = user['role']
            return jsonify({'ok': True, 'role': user['role']})
        return jsonify({'ok': False, 'msg': 'Invalid credentials'}), 401
    return render_template('login.html')


@app.route('/register', methods=['POST'])
def register():
    data  = request.get_json()
    uname = data.get('username', '').strip()
    pwd   = hashlib.sha256(data.get('password', '').encode()).hexdigest()
    role  = data.get('role', 'client')

    # Only client and lawyer can self-register
    # admin and chatbot_manager are created by admin only
    if role not in ('client', 'lawyer'):
        return jsonify({'ok': False, 'msg': 'Invalid role. Only client or lawyer can register.'}), 403

    if not uname or len(uname) < 3:
        return jsonify({'ok': False, 'msg': 'Username must be at least 3 characters'}), 400

    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users(username,password,role) VALUES(?,?,?)",
                (uname, pwd, role)
            )
            conn.commit()
        return jsonify({'ok': True})
    except sqlite3.IntegrityError:
        return jsonify({'ok': False, 'msg': 'Username already exists'}), 409


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ══════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template(
        'dashboard.html',
        username=session['username'],
        role=session['role']
    )


# ══════════════════════════════════════════════════════════════════
#  API: CHATBOT / ANALYZE
# ══════════════════════════════════════════════════════════════════
@app.route('/api/analyze', methods=['POST'])
def analyze():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401

    data = request.get_json()
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    # ── Load conversation history from DB (server-side, unlimited size) ─────────
    import json as _json
    user_id = session.get('user_id')
    conversation_history = []
    try:
        with get_db() as conn:
            row = conn.execute(
                "SELECT history_json FROM chat_sessions WHERE user_id=?", (user_id,)
            ).fetchone()
            if row:
                conversation_history = _json.loads(row['history_json']) or []
    except Exception:
        conversation_history = []

    # Get intelligent legal response (context-aware: intent + full conversation history)
    context_intent_idx = session.get('chat_context_intent_idx')
    result = get_legal_response(
        text,
        context_intent_idx=context_intent_idx,
        conversation_history=conversation_history
    )

    # Unpack 3-tuple: (html, used_ai, matched_intent_idx)
    if isinstance(result, tuple) and len(result) == 3:
        chat_html, used_ai, new_intent_idx = result
    elif isinstance(result, tuple) and len(result) == 2:
        chat_html, used_ai = result
        new_intent_idx = None
    else:
        chat_html, used_ai, new_intent_idx = str(result), False, None

    # ── Save updated conversation history to DB ──────────────────────────────────
    # Strip HTML tags before storing to save space
    plain_response = re.sub(r'<[^>]+>', ' ', chat_html).strip()
    plain_response = re.sub(r'\s+', ' ', plain_response)[:1000]  # cap per message

    conversation_history.append({"role": "user",      "content": text})
    conversation_history.append({"role": "assistant", "content": plain_response})

    # Keep last 20 messages (10 turns) in DB
    if len(conversation_history) > 20:
        conversation_history = conversation_history[-20:]

    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO chat_sessions(user_id, history_json, updated_at)
                   VALUES(?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(user_id) DO UPDATE SET
                       history_json=excluded.history_json,
                       updated_at=CURRENT_TIMESTAMP""",
                (user_id, _json.dumps(conversation_history))
            )
            conn.commit()
    except Exception as e:
        print(f"[Chat History Save Error] {e}")

    # Store the matched intent index in session for rule-based follow-up handling
    if new_intent_idx is not None:
        session['chat_context_intent_idx'] = new_intent_idx

    # ML classification for side panel
    ml_data = None
    try:
        from legal_brain import extract_dynamic_case_data
        # Try to dynamically extract case specifics from the chat
        dynamic_data = extract_dynamic_case_data(conversation_history, text)
        if dynamic_data and 'category' in dynamic_data:
            ml_data = {
                'category': dynamic_data.get('category', 'criminal').lower(),
                'cat_confidence': 95,
                'risk_level': dynamic_data.get('risk_level', 'Medium'),
                'risk_confidence': 95,
                'win_probability': dynamic_data.get('win_probability', 60),
                'similar_cases': [],
                'advanced_strategy': {
                    'how_to_do': dynamic_data.get('recommended_steps', []),
                    'what_to_do': "Follow these specific steps closely.",
                    'when_to_do': "Immediately",
                    'ipc_sections': ', '.join(dynamic_data.get('applicable_laws', []))
                }
            }
    except Exception as e:
        print(f"[Dynamic ML Error] {e}")

    # Fallback to static ML model if dynamic fails
    if not ml_data:
        try:
            ml_data = ml_analyze(text)
        except Exception:
            pass

        # Override ML category with rule-based detection when available
        if new_intent_idx is not None and ml_data:
            correct_category = INTENT_TO_CATEGORY.get(new_intent_idx)
            if correct_category:
                ml_data['category'] = correct_category
                ml_data['cat_confidence'] = 95
                risk_lvl = ml_data.get('risk_level', 'Medium')
                WIN_MAP = {'Low': 85.0, 'Medium': 68.0, 'High': 52.0}
                ml_data['win_probability'] = WIN_MAP.get(risk_lvl, 60.0)

    response = {
        'type'      : 'legal_analysis',
        'message'   : chat_html,
        'used_ai'   : used_ai,
        'disclaimer': 'This system provides informational guidance only, not legal advice. Always consult a qualified lawyer.',
    }

    # ML results — always show if we have category prediction
    # (advanced_strategy how_to_do may be empty if dataset entry is bare)
    if ml_data and ml_data.get('category'):
        # Build strategy: prefer RAG-matched dataset entry, else template
        adv = ml_data.get('advanced_strategy')
        if adv and adv.get('how_to_do'):
            laws_raw = adv.get('ipc_sections', '')
            applicable_laws = [l.strip() for l in re.split(r'[|,]', laws_raw) if l.strip()]
            strategy = {
                'steps'          : adv['how_to_do'],
                'advice'         : f"{adv.get('what_to_do','')}  <em>({adv.get('when_to_do','')})</em>",
                'applicable_laws': applicable_laws or LAW_MAP.get(ml_data['category'], []),
            }
        else:
            # Fallback: use our hardcoded strategy templates (always reliable)
            strategy = generate_strategy(ml_data['category'], ml_data['risk_level'])

        response.update({
            'has_ml'         : True,
            'category'       : ml_data['category'],
            'cat_confidence' : ml_data['cat_confidence'],
            'risk_level'     : ml_data['risk_level'],
            'risk_confidence': ml_data['risk_confidence'],
            'outcome'        : 'win',
            'win_probability': ml_data['win_probability'],
            'similar_cases'  : ml_data['similar_cases'],
            'strategy'       : strategy,
        })
    else:
        response['has_ml'] = False

    # Log this conversation
    try:
        with get_db() as conn:
            conn.execute(
                """INSERT INTO chatbot_logs(user_id,username,query,response,used_ai,category,risk_level)
                   VALUES(?,?,?,?,?,?,?)""",
                (
                    session.get('user_id'),
                    session.get('username', 'anonymous'),
                    text[:500],
                    chat_html[:1000],
                    1 if used_ai else 0,
                    response.get('category', ''),
                    response.get('risk_level', ''),
                )
            )
            conn.commit()
    except Exception:
        pass  # logging failure should not break the response

    return jsonify(response)


# ══════════════════════════════════════════════════════════════════
#  API: CASES (Client & Lawyer/Admin)
# ══════════════════════════════════════════════════════════════════
@app.route('/api/cases', methods=['GET', 'POST'])
@login_required
def cases():
    if request.method == 'POST':
        # Only clients can submit new cases
        if session.get('role') != 'client':
            return jsonify({'error': 'Only clients can submit cases'}), 403

        data  = request.get_json()
        title = data.get('title', '').strip()
        desc  = data.get('description', '').strip()
        if not title or not desc:
            return jsonify({'error': 'Title and description required'}), 400

        ml       = ml_analyze(desc)

        # Apply rule-based category override (ML model is often inaccurate)
        from legal_brain import get_rule_based_answer
        _, intent_idx = get_rule_based_answer(title + ' ' + desc)
        if intent_idx is not None:
            correct_cat = INTENT_TO_CATEGORY.get(intent_idx)
            if correct_cat:
                ml['category'] = correct_cat
                ml['cat_confidence'] = 95   # rule-based is highly reliable
                WIN_MAP = {'Low': 85.0, 'Medium': 68.0, 'High': 52.0}
                ml['win_probability'] = WIN_MAP.get(ml['risk_level'], 60.0)

        strategy = generate_strategy(ml['category'], ml['risk_level'])

        with get_db() as conn:
            cur = conn.execute(
                """INSERT INTO cases(user_id,title,description,category,risk_level,outcome_pred,outcome_prob)
                   VALUES(?,?,?,?,?,?,?)""",
                (session['user_id'], title, desc,
                 ml['category'], ml['risk_level'], 'win', ml['win_probability'])
            )
            conn.commit()
            case_id = cur.lastrowid

        return jsonify({
            'case_id'        : case_id,
            'category'       : ml['category'],
            'cat_confidence' : ml['cat_confidence'],
            'risk_level'     : ml['risk_level'],
            'risk_confidence': ml['risk_confidence'],
            'outcome'        : 'win',
            'win_probability': ml['win_probability'],
            'similar_cases'  : ml['similar_cases'],
            'strategy'       : strategy,
            'disclaimer'     : 'Informational only – not legal advice.',
        })

    # GET — return cases based on role
    with get_db() as conn:
        if session['role'] in ('lawyer', 'admin'):
            rows = conn.execute(
                "SELECT c.*, u.username FROM cases c JOIN users u ON c.user_id=u.id ORDER BY c.created DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM cases WHERE user_id=? ORDER BY created DESC",
                (session['user_id'],)
            ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/cases/<int:case_id>/document', methods=['GET'])
@login_required
def get_document(case_id):
    with get_db() as conn:
        case = conn.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    if not case:
        return jsonify({'error': 'Case not found'}), 404
    doc = generate_document(
        case['title'], case['category'], case['risk_level'],
        case['outcome_pred'], case['description']
    )
    return jsonify({'document': doc})


@app.route('/api/cases/<int:case_id>/messages', methods=['GET', 'POST'])
@login_required
def messages(case_id):
    if request.method == 'POST':
        content = request.get_json().get('content', '').strip()
        if not content:
            return jsonify({'error': 'Empty message'}), 400
        with get_db() as conn:
            conn.execute(
                "INSERT INTO messages(case_id,sender,role,content) VALUES(?,?,?,?)",
                (case_id, session['username'], session['role'], content)
            )
            conn.commit()
        return jsonify({'ok': True})
    with get_db() as conn:
        msgs = conn.execute(
            "SELECT * FROM messages WHERE case_id=? ORDER BY created",
            (case_id,)
        ).fetchall()
    return jsonify([dict(m) for m in msgs])


@app.route('/api/cases/<int:case_id>/status', methods=['PUT'])
@role_required('lawyer', 'admin')
def update_status(case_id):
    status = request.get_json().get('status', 'reviewed')
    with get_db() as conn:
        conn.execute("UPDATE cases SET status=? WHERE id=?", (status, case_id))
        conn.commit()
    return jsonify({'ok': True})


# ══════════════════════════════════════════════════════════════════
#  API: LAWYER NOTES
# ══════════════════════════════════════════════════════════════════
@app.route('/api/cases/<int:case_id>/notes', methods=['GET', 'POST'])
@role_required('lawyer', 'admin')
def case_notes(case_id):
    if request.method == 'POST':
        note = request.get_json().get('note', '').strip()
        if not note:
            return jsonify({'error': 'Note cannot be empty'}), 400
        with get_db() as conn:
            conn.execute(
                "INSERT INTO lawyer_notes(case_id,lawyer_id,note) VALUES(?,?,?)",
                (case_id, session['user_id'], note)
            )
            conn.commit()
        return jsonify({'ok': True})

    with get_db() as conn:
        rows = conn.execute(
            """SELECT ln.*, u.username as lawyer_name
               FROM lawyer_notes ln
               JOIN users u ON ln.lawyer_id=u.id
               WHERE ln.case_id=?
               ORDER BY ln.created DESC""",
            (case_id,)
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ══════════════════════════════════════════════════════════════════
#  API: APPOINTMENTS (Client books, Lawyer/Admin sees)
# ══════════════════════════════════════════════════════════════════
@app.route('/api/book_appointment', methods=['POST'])
@role_required('client')
def book_appointment():
    data     = request.get_json()
    category = data.get('category', 'general')
    note     = data.get('note', '')
    with get_db() as conn:
        lawyer = conn.execute("SELECT id FROM users WHERE role='lawyer' LIMIT 1").fetchone()
        lawyer_id = lawyer['id'] if lawyer else None
        conn.execute(
            "INSERT INTO appointments(client_id,lawyer_id,category,note) VALUES(?,?,?,?)",
            (session['user_id'], lawyer_id, category, note)
        )
        conn.commit()
    return jsonify({
        'ok': True,
        'message': f"Appointment requested for {category.replace('_',' ').title()} Law. A specialist lawyer will contact you shortly."
    })


@app.route('/api/appointments', methods=['GET'])
@login_required
def get_appointments():
    with get_db() as conn:
        if session['role'] in ('lawyer', 'admin'):
            rows = conn.execute(
                """SELECT a.*, u.username as client_name
                   FROM appointments a JOIN users u ON a.client_id=u.id
                   ORDER BY a.created DESC"""
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM appointments WHERE client_id=? ORDER BY created DESC",
                (session['user_id'],)
            ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/appointments/<int:appt_id>/status', methods=['PUT'])
@role_required('lawyer', 'admin')
def update_appointment_status(appt_id):
    """Lawyer can accept or decline a client's appointment request."""
    status = request.get_json().get('status', 'accepted')
    if status not in ('accepted', 'declined', 'completed'):
        return jsonify({'ok': False, 'msg': 'Invalid status'}), 400
    with get_db() as conn:
        conn.execute("UPDATE appointments SET status=? WHERE id=?", (status, appt_id))
        conn.commit()
    return jsonify({'ok': True, 'msg': f'Appointment {status}'})


@app.route('/api/admin/users', methods=['GET'])
@role_required('admin')
def admin_users():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, role, email, created FROM users ORDER BY created DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/admin/create_user', methods=['POST'])
@role_required('admin')
def admin_create_user():
    """Admin can create any role including admin and chatbot_manager."""
    data  = request.get_json()
    uname = data.get('username', '').strip()
    pwd   = hashlib.sha256(data.get('password', '').encode()).hexdigest()
    role  = data.get('role', 'client')

    valid_roles = ('admin', 'lawyer', 'client', 'chatbot_manager')
    if role not in valid_roles:
        return jsonify({'ok': False, 'msg': f'Invalid role. Must be one of: {valid_roles}'}), 400

    if not uname or len(uname) < 3:
        return jsonify({'ok': False, 'msg': 'Username must be at least 3 characters'}), 400

    try:
        with get_db() as conn:
            conn.execute(
                "INSERT INTO users(username,password,role) VALUES(?,?,?)",
                (uname, pwd, role)
            )
            conn.commit()
        return jsonify({'ok': True, 'msg': f'User {uname} created as {role}'})
    except sqlite3.IntegrityError:
        return jsonify({'ok': False, 'msg': 'Username already exists'}), 409


@app.route('/api/admin/users/<int:uid>/role', methods=['PATCH'])
@role_required('admin')
def admin_change_role(uid):
    """Admin can change any user's role."""
    new_role = request.get_json().get('role', '')
    valid_roles = ('admin', 'lawyer', 'client', 'chatbot_manager')
    if new_role not in valid_roles:
        return jsonify({'ok': False, 'msg': 'Invalid role'}), 400
    # Prevent admin from changing their own role (accidental lockout)
    if uid == session['user_id']:
        return jsonify({'ok': False, 'msg': "Cannot change your own role"}), 400
    with get_db() as conn:
        conn.execute("UPDATE users SET role=? WHERE id=?", (new_role, uid))
        conn.commit()
    return jsonify({'ok': True})


@app.route('/api/admin/users/<int:uid>', methods=['DELETE'])
@role_required('admin')
def admin_delete_user(uid):
    """Admin can delete any user (except themselves)."""
    if uid == session['user_id']:
        return jsonify({'ok': False, 'msg': 'Cannot delete yourself'}), 400
    with get_db() as conn:
        conn.execute("DELETE FROM users WHERE id=?", (uid,))
        conn.commit()
    return jsonify({'ok': True})


@app.route('/api/admin/stats', methods=['GET'])
@role_required('admin')
def admin_stats():
    with get_db() as conn:
        total_users    = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        total_cases    = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        pending_cases  = conn.execute("SELECT COUNT(*) FROM cases WHERE status='pending'").fetchone()[0]
        reviewed_cases = conn.execute("SELECT COUNT(*) FROM cases WHERE status='reviewed'").fetchone()[0]
        total_chats    = conn.execute("SELECT COUNT(*) FROM chatbot_logs").fetchone()[0]
        ai_chats       = conn.execute("SELECT COUNT(*) FROM chatbot_logs WHERE used_ai=1").fetchone()[0]
        clients        = conn.execute("SELECT COUNT(*) FROM users WHERE role='client'").fetchone()[0]
        lawyers        = conn.execute("SELECT COUNT(*) FROM users WHERE role='lawyer'").fetchone()[0]
        admins         = conn.execute("SELECT COUNT(*) FROM users WHERE role='admin'").fetchone()[0]
        cbms           = conn.execute("SELECT COUNT(*) FROM users WHERE role='chatbot_manager'").fetchone()[0]

    return jsonify({
        'total_users'   : total_users,
        'total_cases'   : total_cases,
        'pending_cases' : pending_cases,
        'reviewed_cases': reviewed_cases,
        'total_chats'   : total_chats,
        'ai_chats'      : ai_chats,
        'clients'       : clients,
        'lawyers'       : lawyers,
        'admins'        : admins,
        'chatbot_managers': cbms,
    })


@app.route('/api/admin/chatbot_logs', methods=['GET'])
@role_required('admin', 'chatbot_manager')
def admin_chatbot_logs():
    limit = request.args.get('limit', 50, type=int)
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM chatbot_logs ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ══════════════════════════════════════════════════════════════════
#  API: CHATBOT MANAGER — Intent Management
# ══════════════════════════════════════════════════════════════════
@app.route('/api/chatbot/intents', methods=['GET'])
@role_required('admin', 'chatbot_manager')
def get_intents():
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM chatbot_intents ORDER BY created DESC"
        ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route('/api/chatbot/intents', methods=['POST'])
@role_required('admin', 'chatbot_manager')
def add_intent():
    data     = request.get_json()
    trigger  = data.get('trigger', '').strip().lower()
    response = data.get('response', '').strip()
    category = data.get('category', 'general')

    if not trigger or not response:
        return jsonify({'ok': False, 'msg': 'Trigger and response are required'}), 400

    with get_db() as conn:
        conn.execute(
            "INSERT INTO chatbot_intents(trigger,response,category) VALUES(?,?,?)",
            (trigger, response, category)
        )
        conn.commit()
    return jsonify({'ok': True, 'msg': f'Intent "{trigger}" added successfully'})


@app.route('/api/chatbot/intents/<int:intent_id>', methods=['DELETE'])
@role_required('admin', 'chatbot_manager')
def delete_intent(intent_id):
    with get_db() as conn:
        conn.execute("DELETE FROM chatbot_intents WHERE id=?", (intent_id,))
        conn.commit()
    return jsonify({'ok': True})


@app.route('/api/chatbot/intents/<int:intent_id>/toggle', methods=['PATCH'])
@role_required('admin', 'chatbot_manager')
def toggle_intent(intent_id):
    with get_db() as conn:
        current = conn.execute(
            "SELECT active FROM chatbot_intents WHERE id=?", (intent_id,)
        ).fetchone()
        if not current:
            return jsonify({'ok': False, 'msg': 'Intent not found'}), 404
        new_val = 0 if current['active'] else 1
        conn.execute(
            "UPDATE chatbot_intents SET active=? WHERE id=?", (new_val, intent_id)
        )
        conn.commit()
    return jsonify({'ok': True, 'active': bool(new_val)})


@app.route('/api/chatbot/stats', methods=['GET'])
@role_required('admin', 'chatbot_manager')
def chatbot_stats():
    with get_db() as conn:
        total   = conn.execute("SELECT COUNT(*) FROM chatbot_logs").fetchone()[0]
        ai_used = conn.execute("SELECT COUNT(*) FROM chatbot_logs WHERE used_ai=1").fetchone()[0]
        top_cats = conn.execute(
            """SELECT category, COUNT(*) as cnt FROM chatbot_logs
               WHERE category != ''
               GROUP BY category ORDER BY cnt DESC LIMIT 7"""
        ).fetchall()
        recent_7d = conn.execute(
            """SELECT COUNT(*) FROM chatbot_logs
               WHERE created >= datetime('now', '-7 days')"""
        ).fetchone()[0]
        intents_count = conn.execute(
            "SELECT COUNT(*) FROM chatbot_intents WHERE active=1"
        ).fetchone()[0]

    return jsonify({
        'total_queries'  : total,
        'ai_used'        : ai_used,
        'rule_based'     : total - ai_used,
        'ai_percent'     : round((ai_used / total * 100) if total > 0 else 0, 1),
        'queries_7d'     : recent_7d,
        'top_categories' : [dict(r) for r in top_cats],
        'active_intents' : intents_count,
    })


# ══════════════════════════════════════════════════════════════════
#  API: SHARED
# ══════════════════════════════════════════════════════════════════
@app.route('/api/me')
def me():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    with get_db() as conn:
        user = conn.execute(
            "SELECT id, username, role, email, full_name FROM users WHERE id=?",
            (session['user_id'],)
        ).fetchone()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(dict(user))


# ══════════════════════════════════════════════════════════════════
#  API: PROFILE UPDATE (name, email)
# ══════════════════════════════════════════════════════════════════
@app.route('/api/profile/update', methods=['POST'])
@login_required
def profile_update():
    data      = request.get_json()
    full_name = data.get('full_name', '').strip()
    email     = data.get('email', '').strip()
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET full_name=?, email=? WHERE id=?",
            (full_name, email, session['user_id'])
        )
        conn.commit()
    return jsonify({'ok': True, 'msg': 'Profile updated successfully'})


# ══════════════════════════════════════════════════════════════════
#  API: NOTIFICATIONS (role-specific)
# ══════════════════════════════════════════════════════════════════
@app.route('/api/notifications')
@login_required
def notifications():
    notifs = []
    role    = session.get('role')
    user_id = session.get('user_id')
    with get_db() as conn:
        if role == 'client':
            cases = conn.execute(
                "SELECT id, title, status, created FROM cases WHERE user_id=? ORDER BY created DESC LIMIT 6",
                (user_id,)
            ).fetchall()
            for c in cases:
                notifs.append({'icon': '📁', 'text': f'Case "{c["title"][:30]}" — {c["status"]}', 'time': c['created'], 'type': c['status']})
            appts = conn.execute(
                "SELECT category, status, created FROM appointments WHERE client_id=? ORDER BY created DESC LIMIT 4",
                (user_id,)
            ).fetchall()
            for a in appts:
                icon = '✅' if a['status']=='accepted' else ('❌' if a['status']=='declined' else '📅')
                notifs.append({'icon': icon, 'text': f'Appointment ({a["category"]}) — {a["status"]}', 'time': a['created'], 'type': a['status']})

        elif role == 'lawyer':
            pending = conn.execute(
                "SELECT id, title, created FROM cases WHERE status='pending' ORDER BY created DESC LIMIT 5"
            ).fetchall()
            for c in pending:
                notifs.append({'icon': '⏳', 'text': f'Pending: "{c["title"][:40]}"', 'time': c['created'], 'type': 'pending'})
            appts = conn.execute(
                """SELECT a.category, a.created, u.username FROM appointments a
                   JOIN users u ON a.client_id=u.id
                   WHERE a.status='requested' ORDER BY a.created DESC LIMIT 4"""
            ).fetchall()
            for a in appts:
                notifs.append({'icon': '📅', 'text': f'{a["username"]} → {a["category"]} appointment', 'time': a['created'], 'type': 'requested'})

        elif role == 'admin':
            recent_users = conn.execute(
                "SELECT username, role, created FROM users ORDER BY created DESC LIMIT 4"
            ).fetchall()
            for u in recent_users:
                notifs.append({'icon': '👤', 'text': f'User: {u["username"]} ({u["role"]})', 'time': u['created'], 'type': 'user'})
            pending = conn.execute("SELECT COUNT(*) FROM cases WHERE status='pending'").fetchone()[0]
            if pending > 0:
                notifs.append({'icon': '⚠️', 'text': f'{pending} cases pending lawyer review', 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'type': 'alert'})
            total_chats = conn.execute("SELECT COUNT(*) FROM chatbot_logs").fetchone()[0]
            notifs.append({'icon': '💬', 'text': f'Total AI queries: {total_chats}', 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'type': 'info'})

        elif role == 'chatbot_manager':
            recent = conn.execute(
                "SELECT username, SUBSTR(query,1,50) as q, created FROM chatbot_logs ORDER BY created DESC LIMIT 6"
            ).fetchall()
            for r in recent:
                notifs.append({'icon': '💬', 'text': f'{r["username"]}: {r["q"]}', 'time': r['created'], 'type': 'chat'})
            intents = conn.execute("SELECT COUNT(*) FROM chatbot_intents WHERE active=1").fetchone()[0]
            notifs.append({'icon': '🤖', 'text': f'{intents} active custom intents', 'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'type': 'info'})

    notifs.sort(key=lambda x: x['time'], reverse=True)
    return jsonify(notifs[:8])


# ══════════════════════════════════════════════════════════════════
#  API: CLIENT — VIEW OWN CASE DETAILS (including lawyer notes)
# ══════════════════════════════════════════════════════════════════
@app.route('/api/cases/<int:case_id>/client_view')
@role_required('client')
def case_client_view(case_id):
    with get_db() as conn:
        case = conn.execute(
            "SELECT * FROM cases WHERE id=? AND user_id=?",
            (case_id, session['user_id'])
        ).fetchone()
        if not case:
            return jsonify({'error': 'Case not found'}), 404
        notes = conn.execute(
            """SELECT ln.note, ln.created, u.username as lawyer_name
               FROM lawyer_notes ln JOIN users u ON ln.lawyer_id=u.id
               WHERE ln.case_id=? ORDER BY ln.created DESC""",
            (case_id,)
        ).fetchall()
        msgs = conn.execute(
            "SELECT * FROM messages WHERE case_id=? ORDER BY created",
            (case_id,)
        ).fetchall()
    return jsonify({
        'case'    : dict(case),
        'notes'   : [dict(n) for n in notes],
        'messages': [dict(m) for m in msgs],
    })


@app.route('/api/change_password', methods=['POST'])
@login_required
def change_password():
    data       = request.get_json()
    current_pw = hashlib.sha256(data.get('current_password', '').encode()).hexdigest()
    new_pw     = hashlib.sha256(data.get('new_password', '').encode()).hexdigest()
    with get_db() as conn:
        user = conn.execute(
            "SELECT id FROM users WHERE id=? AND password=?",
            (session['user_id'], current_pw)
        ).fetchone()
        if not user:
            return jsonify({'ok': False, 'msg': 'Current password is incorrect'}), 401
        conn.execute(
            "UPDATE users SET password=? WHERE id=?", (new_pw, session['user_id'])
        )
        conn.commit()
    return jsonify({'ok': True})



# ══════════════════════════════════════════════════════════════════
#  API: CLIENT STATS
# ══════════════════════════════════════════════════════════════════
@app.route('/api/client/stats')
@role_required('client')
def client_stats():
    with get_db() as conn:
        cases = conn.execute(
            "SELECT status FROM cases WHERE user_id=?", (session['user_id'],)
        ).fetchall()
        appts = conn.execute(
            "SELECT status FROM appointments WHERE client_id=?", (session['user_id'],)
        ).fetchall()
    total    = len(cases)
    pending  = sum(1 for c in cases if c['status'] == 'pending')
    reviewed = sum(1 for c in cases if c['status'] == 'reviewed')
    closed   = sum(1 for c in cases if c['status'] == 'closed')
    return jsonify({
        'total_cases'    : total,
        'pending'        : pending,
        'reviewed'       : reviewed,
        'closed'         : closed,
        'total_appts'    : len(appts),
        'appts_pending'  : sum(1 for a in appts if a['status'] == 'requested'),
        'appts_accepted' : sum(1 for a in appts if a['status'] == 'accepted'),
    })


# ══════════════════════════════════════════════════════════════════
#  API: LAWYER STATS
# ══════════════════════════════════════════════════════════════════
@app.route('/api/lawyer/stats')
@role_required('lawyer', 'admin')
def lawyer_stats():
    with get_db() as conn:
        total         = conn.execute("SELECT COUNT(*) FROM cases").fetchone()[0]
        pending       = conn.execute("SELECT COUNT(*) FROM cases WHERE status='pending'").fetchone()[0]
        reviewed      = conn.execute("SELECT COUNT(*) FROM cases WHERE status='reviewed'").fetchone()[0]
        appts_pending = conn.execute("SELECT COUNT(*) FROM appointments WHERE status='requested'").fetchone()[0]
        today_rev     = conn.execute(
            "SELECT COUNT(*) FROM lawyer_notes WHERE DATE(created)=DATE('now') AND lawyer_id=?",
            (session['user_id'],)
        ).fetchone()[0]
        my_notes_total = conn.execute(
            "SELECT COUNT(*) FROM lawyer_notes WHERE lawyer_id=?", (session['user_id'],)
        ).fetchone()[0]
    return jsonify({
        'total'          : total,
        'pending'        : pending,
        'reviewed'       : reviewed,
        'appts_pending'  : appts_pending,
        'today_reviewed' : today_rev,
        'my_notes_total' : my_notes_total,
    })


# ══════════════════════════════════════════════════════════════════
#  API: CANCEL CASE (Client — pending only)
# ══════════════════════════════════════════════════════════════════
@app.route('/api/cases/<int:case_id>/cancel', methods=['DELETE'])
@role_required('client')
def cancel_case(case_id):
    with get_db() as conn:
        case = conn.execute(
            "SELECT * FROM cases WHERE id=? AND user_id=?",
            (case_id, session['user_id'])
        ).fetchone()
        if not case:
            return jsonify({'ok': False, 'msg': 'Case not found or not yours'}), 404
        if case['status'] != 'pending':
            return jsonify({'ok': False, 'msg': 'Only pending cases can be cancelled'}), 400
        conn.execute("UPDATE cases SET status='closed' WHERE id=?", (case_id,))
        conn.commit()
    return jsonify({'ok': True, 'msg': 'Case cancelled successfully'})


# ══════════════════════════════════════════════════════════════════
#  API: CANCEL APPOINTMENT (Client)
# ══════════════════════════════════════════════════════════════════
@app.route('/api/appointments/<int:appt_id>/cancel', methods=['DELETE'])
@role_required('client')
def cancel_appointment(appt_id):
    with get_db() as conn:
        appt = conn.execute(
            "SELECT * FROM appointments WHERE id=? AND client_id=?",
            (appt_id, session['user_id'])
        ).fetchone()
        if not appt:
            return jsonify({'ok': False, 'msg': 'Appointment not found or not yours'}), 404
        if appt['status'] in ('completed', 'declined'):
            return jsonify({'ok': False, 'msg': f'Cannot cancel a {appt["status"]} appointment'}), 400
        conn.execute("UPDATE appointments SET status='cancelled' WHERE id=?", (appt_id,))
        conn.commit()
    return jsonify({'ok': True, 'msg': 'Appointment cancelled successfully'})


# ══════════════════════════════════════════════════════════════════
#  API: ADMIN - SYSTEM CONFIG
# ══════════════════════════════════════════════════════════════════
@app.route('/api/admin/config', methods=['GET', 'POST'])
@role_required('admin', 'chatbot_manager')
def admin_config():
    if request.method == 'POST':
        data = request.get_json()
        with get_db() as conn:
            for key, val in data.items():
                conn.execute(
                    "INSERT OR REPLACE INTO system_config(key,value) VALUES(?,?)",
                    (key, str(val))
                )
            conn.commit()
        return jsonify({'ok': True, 'msg': 'Config updated'})
    with get_db() as conn:
        rows = conn.execute("SELECT key, value FROM system_config").fetchall()
    return jsonify({r['key']: r['value'] for r in rows})


# ══════════════════════════════════════════════════════════════════
#  API: ADMIN - RECENT ACTIVITY FEED
# ══════════════════════════════════════════════════════════════════
@app.route('/api/admin/activity')
@role_required('admin')
def admin_activity():
    with get_db() as conn:
        cases = conn.execute(
            """SELECT 'case_submitted' as type, c.title as detail, u.username, c.created, c.status
               FROM cases c JOIN users u ON c.user_id=u.id
               ORDER BY c.created DESC LIMIT 6"""
        ).fetchall()
        appts = conn.execute(
            """SELECT 'appointment' as type, a.category as detail, u.username, a.created, a.status
               FROM appointments a JOIN users u ON a.client_id=u.id
               ORDER BY a.created DESC LIMIT 4"""
        ).fetchall()
        chats = conn.execute(
            """SELECT 'chat_query' as type, SUBSTR(query,1,60) as detail, username, created, '' as status
               FROM chatbot_logs ORDER BY created DESC LIMIT 5"""
        ).fetchall()
    all_activity = [dict(r) for r in list(cases) + list(appts) + list(chats)]
    all_activity.sort(key=lambda x: x['created'], reverse=True)
    return jsonify(all_activity[:12])


# ══════════════════════════════════════════════════════════════════
#  API: LAWYER - MY NOTES
# ══════════════════════════════════════════════════════════════════
@app.route('/api/lawyer/my_notes')
@role_required('lawyer', 'admin')
def lawyer_my_notes():
    with get_db() as conn:
        rows = conn.execute(
            """SELECT ln.*, c.title as case_title, c.category as case_category
               FROM lawyer_notes ln
               JOIN cases c ON ln.case_id=c.id
               WHERE ln.lawyer_id=?
               ORDER BY ln.created DESC""",
            (session['user_id'],)
        ).fetchall()
    return jsonify([dict(r) for r in rows])


# ══════════════════════════════════════════════════════════════════
#  API: CHATBOT MANAGER - BULK INTENTS
# ══════════════════════════════════════════════════════════════════
@app.route('/api/chatbot/bulk_intents', methods=['POST'])
@role_required('admin', 'chatbot_manager')
def bulk_intents():
    data    = request.get_json()
    intents = data.get('intents', [])
    added   = 0
    with get_db() as conn:
        for intent in intents:
            trigger  = intent.get('trigger', '').strip().lower()
            response = intent.get('response', '').strip()
            category = intent.get('category', 'general')
            if trigger and response:
                try:
                    conn.execute(
                        "INSERT INTO chatbot_intents(trigger,response,category) VALUES(?,?,?)",
                        (trigger, response, category)
                    )
                    added += 1
                except Exception:
                    pass
        conn.commit()
    return jsonify({'ok': True, 'added': added, 'msg': f'{added} intents added successfully'})


# ══════════════════════════════════════════════════════════════════
#  ERROR HANDLERS
# ══════════════════════════════════════════════════════════════════
@app.errorhandler(404)
def not_found(e):
    return render_template('error.html', code=404, title='Page Not Found',
        message='The page you are looking for does not exist.'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', code=500, title='Server Error',
        message='Something went wrong. Please try again.'), 500


@app.errorhandler(403)
def forbidden(e):
    return render_template('error.html', code=403, title='Access Forbidden',
        message='You do not have permission to access this resource.'), 403


# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
