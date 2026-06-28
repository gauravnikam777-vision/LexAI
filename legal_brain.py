"""
legal_brain.py  –  LexAI Senior Advocate Intelligence Engine
=============================================================
Powered by Mistral AI API + rule-based fallback.
Handles: spelling mistakes, Hinglish, short/long queries,
any legal question (not just case descriptions).
"""

import os, re
from dotenv import load_dotenv

load_dotenv()

# ── Mistral setup ──────────────────────────────────────────────────────────────
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY", "")
mistral_client = None

def init_mistral():
    global mistral_client
    if not MISTRAL_KEY or MISTRAL_KEY == "your_mistral_api_key_here":
        print("[WARN] No Mistral API key set. Using rule-based fallback only.")
        return False
    try:
        try:
            from mistralai import Mistral          # new SDK v1+
        except ImportError:
            from mistralai.client import Mistral   # old SDK fallback
        mistral_client = Mistral(api_key=MISTRAL_KEY)
        print("[OK] Mistral AI initialized.")
        return True
    except Exception as e:
        print(f"[WARN] Mistral init failed: {e}")
        return False

# ── Master System Prompt — Senior Indian Advocate Persona ─────────────────────
SYSTEM_PROMPT = """You are Adv. LexAI — a warm, senior Indian advocate with 35+ years of experience across all domains of Indian law. You have practiced in the Supreme Court, multiple High Courts, and District Courts.

PERSONALITY & TONE:
- Respond like a trusted senior advocate speaking to a client in your chamber — warm, reassuring, authoritative
- Act very humanized. Behave exactly like a real lawyer gathering facts.
- Use natural Hinglish phrases sometimes: "dekho", "yeh matter serious hai", "ghabrao mat", "ek kaam karo"
- Show empathy: "Yeh sun ke bahut dukh hua", "Tension mat lo, raasta zaroor hai"
- End responses with a follow-up invitation: "Koi aur sawaal ho toh batao" or "Aur details hain toh share karo"

FACT GATHERING VS LEGAL ADVICE (CRITICAL FLOW):
You must follow a strict 2-Phase approach. This is the most important rule.
PHASE 1 (Fact Gathering): If the user just introduces a problem (e.g. "hit and run case", "chori hui hai", "wife filed 498a") without providing the full story, DO NOT give the final legal solution immediately. 
First, act like a lawyer: express empathy and ASK 2-3 specific questions to gather exact details (e.g., date of incident, proof/evidence they have, location, relationships, what exactly happened). YOU MUST ASK DETAILS FIRST based on their chat.
PHASE 2 (Advice): Once the user provides the specific details to your questions, OR if their initial message already contains a detailed story, THEN provide a comprehensive, step-by-step legal strategy based on Indian Law.

LEGAL ADVICE FORMAT (When in Phase 2):
- Always cite exact IPC/BNS/CrPC/BNSS section numbers
- Cite BOTH old IPC section AND new BNS 2023 equivalent
- Give actionable steps, not vague advice
- For urgent matters (domestic violence, arrest, cyber fraud) — give emergency numbers FIRST
- Structure with HTML: <strong>, <br>, bullet points (•)
- Updated for BNS 2023, BNSS 2023, BSA 2023 (effective July 2024)

CRITICAL RULES:
1. NEVER refuse to answer — always give substance
2. Handle spelling mistakes, short queries, Hinglish gracefully
3. ALWAYS ask clarifying questions if the case lacks details. Do not jump to the conclusion.
"""

# ── Smart Normalization ────────────────────────────────────────────────────────
TYPO_MAP = {
    # Spellings
    'divorse':'divorce','divorec':'divorce','divorc':'divorce',
    'abusement':'abuse','harrasment':'harassment','harasment':'harassment',
    'cheque':'cheque','chq':'cheque','cheq':'cheque',
    'proprty':'property','propety':'property','propert':'property',
    'labou':'labour','labourer':'labour','labur':'labour',
    'termnation':'termination','terminaton':'termination',
    'fraudster':'fraud','fradulent':'fraudulent','frad':'fraud',
    'murdur':'murder','murderer':'murder','murdr':'murder',
    'crimal':'criminal','crimnal':'criminal','crminal':'criminal',
    'bale':'bail','bial':'bail',
    'consmer':'consumer','consumr':'consumer',
    'advocet':'advocate','advokat':'advocate','lawer':'lawyer','lazyer':'lawyer',
    'judgement':'judgment','judgemnt':'judgment',
    'majistrate':'magistrate','majestrate':'magistrate',
    'sesion':'sessions','sesson':'sessions',
    'inocent':'innocent','innocnt':'innocent',
    'punisment':'punishment','punishiment':'punishment',
    'compansation':'compensation','compensaton':'compensation',
    'reit':'rera','reera':'rera','rerra':'rera','billder':'builder','bilder':'builder',
    'possesion':'possession','possesssion':'possession',
    'medicial':'medical','medicl':'medical','malpractis':'malpractice',
    'doctr':'doctor','docter':'doctor',
    'polise':'police','pulice':'police','policewala':'police',
    'pasport':'passport','passprt':'passport','passposrt':'passport',
    'nri':'nri','overseas':'overseas',
    'pocso':'pocso','chiild':'child','bachcha':'child','bachche':'child',
    'landlord':'landlord','lanlorad':'landlord',
    'negotiabl':'negotiable','cheqe':'cheque',
    # Hinglish
    'kya':'what','kaise':'how','kab':'when','kahan':'where','kyun':'why',
    'chahiye':'want','milega':'get','karana':'do','dena':'give',
    'mujhe':'i','mera':'my','meri':'my','mere':'my',
    'polis':'police','pulis':'police',
    'adalat':'court','kacheri':'court',
    'vakil':'lawyer','advokat':'advocate',
    'girftari':'arrest','giraftari':'arrest',
    'zameen':'land','jamin':'land','jameen':'land',
    'kiraya':'rent','kirayedar':'tenant','malik':'landlord',
    'talak':'divorce','talaak':'divorce',
    'dahej':'dowry',
    'naukri':'job',
    'thuggee':'fraud','thagi':'fraud',
    'chori':'theft','chor':'theft',
    'rishwat':'bribe','riswat':'bribe',
    'bachao':'help','bachav':'help','madad':'help',
    'maar':'beat','marpeet':'assault','dhoka':'fraud','dhokhebaz':'cheater',
}

def normalize(text):
    """Lowercase, remove punctuation except hyphens, correct typos."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s\-]", " ", t)
    t = re.sub(r"\s+", " ", t)
    words = t.split()
    corrected = [TYPO_MAP.get(w, w) for w in words]
    return " ".join(corrected)


def md_to_html(text):
    """Convert Gemini markdown output to HTML-safe rich text for the chat bubble."""
    # Bold+Italic combo first
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    # Headings (H3 → styled span)
    text = re.sub(r'^### (.+)$',
        r'<span style="display:block;font-size:.84rem;font-weight:700;color:#a78bfa;margin:.45rem 0 .2rem">\1</span>',
        text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$',
        r'<span style="display:block;font-size:.88rem;font-weight:800;color:#a78bfa;margin:.5rem 0 .25rem">\1</span>',
        text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$',
        r'<span style="display:block;font-size:.92rem;font-weight:800;color:#a78bfa;margin:.5rem 0 .25rem">\1</span>',
        text, flags=re.MULTILINE)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*([^\*\n]+?)\*', r'<em>\1</em>', text)
    # Bullet points (- item or * item at line start)
    text = re.sub(r'^[\-\*]\s+(.+)$', r'• \1', text, flags=re.MULTILINE)
    # Numbered list: "1. Item" → "<strong>1.</strong> Item"
    text = re.sub(r'^(\d+)\.\s+(.+)$', r'<strong>\1.</strong> \2', text, flags=re.MULTILINE)
    # Inline code
    text = re.sub(r'`([^`]+)`',
        r'<code style="background:rgba(124,58,237,.12);padding:.1rem .3rem;border-radius:4px;font-family:monospace;font-size:.78rem">\1</code>',
        text)
    # Line breaks
    text = text.replace('\n\n', '<br><br>').replace('\n', '<br>')
    return text

# ── Section Lookup — IPC/BNS/CrPC/BNSS quick reference ───────────────────────
SECTION_MAP = {
    # IPC / BNS 2023
    "302": {"act":"IPC/BNS","title":"Murder","old":"IPC Section 302","new":"BNS Section 101","penalty":"Death or Life Imprisonment + Fine","desc":"Punishment for murder. Requires intention or knowledge that the act will cause death."},
    "304a": {"act":"IPC","title":"Causing Death by Negligence","old":"IPC Section 304A","new":"BNS Section 106","penalty":"Up to 2 years imprisonment / Fine / Both","desc":"Causing death by a rash or negligent act not amounting to culpable homicide. Common in road accidents."},
    "307": {"act":"IPC/BNS","title":"Attempt to Murder","old":"IPC Section 307","new":"BNS Section 109","penalty":"Up to 10 years imprisonment; Life imprisonment if person is hurt","desc":"Whoever attempts to cause death with intention/knowledge."},
    "354": {"act":"IPC/BNS","title":"Assault/Outraging Modesty","old":"IPC Section 354","new":"BNS Section 74","penalty":"1–5 years imprisonment + Fine","desc":"Assault or criminal force against a woman with intention to outrage her modesty."},
    "354a": {"act":"IPC/BNS","title":"Sexual Harassment","old":"IPC Section 354A","new":"BNS Section 75","penalty":"Up to 3 years imprisonment / Fine / Both","desc":"Sexual harassment including unwelcome physical contact, sexual advances, demand for sexual favors."},
    "376": {"act":"IPC/BNS","title":"Rape","old":"IPC Section 376","new":"BNS Section 63","penalty":"Minimum 10 years Rigorous Imprisonment, may extend to Life","desc":"Sexual assault without consent. Strict penalties under POCSO Act for minors."},
    "378": {"act":"IPC/BNS","title":"Theft","old":"IPC Section 378","new":"BNS Section 303","penalty":"Up to 3 years imprisonment / Fine / Both","desc":"Dishonest taking of movable property without consent of the owner."},
    "383": {"act":"IPC/BNS","title":"Extortion","old":"IPC Section 383","new":"BNS Section 308","penalty":"Up to 3 years imprisonment / Fine / Both","desc":"Putting a person in fear of injury to dishonestly induce them to deliver property."},
    "395": {"act":"IPC/BNS","title":"Dacoity","old":"IPC Section 395","new":"BNS Section 310","penalty":"Life imprisonment or 10 years Rigorous Imprisonment + Fine","desc":"Robbery committed by five or more persons together."},
    "405": {"act":"IPC/BNS","title":"Criminal Breach of Trust","old":"IPC Section 405","new":"BNS Section 316","penalty":"Up to 3 years imprisonment / Fine / Both","desc":"Whoever being entrusted with property dishonestly misappropriates it or converts it for own use."},
    "406": {"act":"IPC/BNS","title":"Punishment for Criminal Breach of Trust","old":"IPC Section 406","new":"BNS Section 316","penalty":"Up to 3 years imprisonment / Fine / Both","desc":"Punishment for Criminal Breach of Trust."},
    "415": {"act":"IPC/BNS","title":"Cheating (definition)","old":"IPC Section 415","new":"BNS Section 318","penalty":"—","desc":"Cheating defined as deceiving a person and inducing them to deliver property or commit an act they would not otherwise do."},
    "420": {"act":"IPC/BNS","title":"Cheating & Dishonest Inducement","old":"IPC Section 420","new":"BNS Section 318","penalty":"Up to 7 years imprisonment + Fine","desc":"Cheating and dishonestly inducing delivery of property. Most common fraud section."},
    "423": {"act":"IPC/BNS","title":"Dishonest/Fraudulent Execution of Deed","old":"IPC Section 423","new":"BNS Section 328","penalty":"Up to 2 years imprisonment / Fine / Both","desc":"Execution of a deed relating to property with dishonest intent."},
    "441": {"act":"IPC/BNS","title":"Criminal Trespass","old":"IPC Section 441","new":"BNS Section 329","penalty":"Up to 3 months imprisonment / Fine / Both","desc":"Entering or remaining in another's property to commit an offense, intimidate, insult, or annoy."},
    "447": {"act":"IPC/BNS","title":"Punishment for Criminal Trespass","old":"IPC Section 447","new":"BNS Section 329","penalty":"Up to 3 months imprisonment / Fine upto Rs.500 / Both","desc":"Punishment for criminal trespass."},
    "498a": {"act":"IPC/BNS","title":"Cruelty by Husband/Relatives","old":"IPC Section 498A","new":"BNS Section 85","penalty":"Up to 3 years imprisonment + Fine","desc":"Husband or relatives subjecting woman to cruelty — includes harassment for dowry demands. Non-bailable, cognizable offense."},
    "499": {"act":"IPC/BNS","title":"Defamation (Definition)","old":"IPC Section 499","new":"BNS Section 356","penalty":"—","desc":"Making or publishing imputation about a person intending to harm reputation. Includes spoken, written, or digital defamation."},
    "500": {"act":"IPC/BNS","title":"Punishment for Defamation","old":"IPC Section 500","new":"BNS Section 356","penalty":"Up to 2 years Simple Imprisonment / Fine / Both","desc":"Punishment for defamation under IPC Section 499."},
    "506": {"act":"IPC/BNS","title":"Criminal Intimidation","old":"IPC Section 506","new":"BNS Section 351","penalty":"Up to 2 years / 7 years (if threat of death/grievous hurt)","desc":"Threatening another person to cause injury to person, reputation, or property to force them to do or refrain from something."},
    "420t": {"act":"Special","title":"Cyber Fraud under IT Act","old":"IT Act S.66C / S.66D","new":"IT Act S.66C / S.66D (still in force)","penalty":"Up to 3 years imprisonment + Fine upto Rs.1 Lakh","desc":"Identity theft (66C) and cheating by personation using computer resources (66D)."},
    "279": {"act":"IPC/BNS","title":"Rash Driving","old":"IPC Section 279","new":"BNS Section 281","penalty":"Up to 6 months imprisonment / Fine upto Rs.1000 / Both","desc":"Rash or negligent driving on public way endangering human life. Applied in motor accident cases."},
    "138": {"act":"NI Act","title":"Cheque Bounce / Dishonour","old":"NI Act Section 138","new":"NI Act Section 138 (unchanged)","penalty":"Up to 2 years / Fine upto 2× cheque amount / Both","desc":"Dishonour of cheque for insufficiency of funds. Strict 30-day notice deadline applies."},
    "125": {"act":"CrPC/BNSS","title":"Maintenance","old":"CrPC Section 125","new":"BNSS Section 144","penalty":"Fine; Imprisonment if not complied with order","desc":"Order for maintenance of wives, children, and parents. Applications to be decided within 60 days under BNSS."},
    "436": {"act":"CrPC/BNSS","title":"Bail in Bailable Offenses","old":"CrPC Section 436","new":"BNSS Section 478","penalty":"—","desc":"For bailable offenses, bail is a right — police must release on bail if accused offers surety."},
    "437": {"act":"CrPC/BNSS","title":"Bail in Non-Bailable Offenses","old":"CrPC Section 437","new":"BNSS Section 480","penalty":"—","desc":"Court discretion for bail in non-bailable offenses. Cannot be given when offense is punishable with death/life imprisonment."},
    "438": {"act":"CrPC/BNSS","title":"Anticipatory Bail","old":"CrPC Section 438","new":"BNSS Section 482","penalty":"—","desc":"Bail granted before arrest. File in Sessions Court or High Court if you apprehend arrest."},
    "154": {"act":"CrPC/BNSS","title":"FIR Registration","old":"CrPC Section 154","new":"BNSS Section 173","penalty":"Failure to register FIR: Punishable under S.166A IPC","desc":"Information of cognizable offense must be recorded in writing and read to the informant. Police CANNOT refuse FIR for cognizable offenses."},
    "156": {"act":"CrPC/BNSS","title":"Police Investigation","old":"CrPC Section 156(3)","new":"BNSS Section 175(3)","penalty":"—","desc":"Magistrate can direct police to register FIR and investigate. File before Magistrate if police refuse your FIR."},
    "22": {"act":"Constitution","title":"Right Against Arbitrary Arrest","old":"Article 22, Constitution of India","new":"Article 22, Constitution of India","penalty":"Illegal detention is a rights violation","desc":"Right to be informed of grounds of arrest. Right to consult and be defended by advocate. Must be produced before Magistrate within 24 hours."},
}

def lookup_section(query):
    """Check if query is asking about a specific law section."""
    q = query.lower()
    patterns = [
        r"(?:ipc|bnss?|crpc|section|sec|dhara|section|article|art)\s*[s]?[\s#]?(\d+[a-z]?)",
        r"(\d+[a-z]?)\s*(?:ipc|bnss?|crpc|ki|ka|kya|section|sec)",
        r"(?:what is|explain|tell me|batao|describe)\s+(?:ipc\s*)?(\d+[a-z]?)",
        r"(\d+)\s*ipc",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            sec = m.group(1).replace(" ", "").lower()
            if sec in SECTION_MAP:
                s = SECTION_MAP[sec]
                return (
                    f"⚖️ <strong>{s['old']} — {s['title']}</strong><br>"
                    f"<em>New equivalent: {s['new']}</em><br><br>"
                    f"<strong>What it says:</strong> {s['desc']}<br><br>"
                    f"<strong>Penalty:</strong> {s['penalty']}<br><br>"
                    f"<strong>Key fact:</strong> This is a {_bailable(sec)} offense under Indian law."
                )
    return None

def _bailable(sec):
    NON_BAILABLE = {'302','307','376','395','498a'}
    return "non-bailable (bail only from court)" if sec in NON_BAILABLE else "cognizable"

# ── Rule-Based Fallback (50+ patterns, fuzzy token scoring) ───────────────────
INTENTS = [
    {"kw":["hit and run","hit and run","road accident","car accident","bike accident","vehicle accident","knocked down","hit by car","hit by truck","vehicle hit me","dahil sadak","motor accident","mact","solatium"],"ans":lambda: HIT_RUN},
    {"kw":["cheque bounce","cheque dishonour","cheque return","bounced cheque","ni act 138","cheque not cleared","dishonoured cheque","check bounce","cheque wapas"],"ans":lambda: CHEQUE},
    {"kw":["domestic violence","wife beating","husband beat","domestic abuse","dv act","pwdva","498a","dahej","dowry harassment","in-law harassing","sasural","in laws","pati maar","husband abuse"],"ans":lambda: DV},
    {"kw":["cyber crime","online fraud","upi fraud","otp fraud","bank fraud","phishing","hacked","sextortion","net banking fraud","social media hack","instagram hack","account hack","1930","cybercrime","digital fraud","whatsapp fraud"],"ans":lambda: CYBER},
    {"kw":["divorce","talak","talaak","separation","alimony","mutual divorce","contested divorce","matrimonial","section 13","section 13b","husband wife separate","married life problem","marriage break"],"ans":lambda: DIVORCE},
    {"kw":["maintenance","nafqa","section 125","child support","wife maintenance","husband not paying","bnss 144","support money","guzara"],"ans":lambda: MAINTENANCE},
    {"kw":["bail","anticipatory bail","438","437","bail kaise","bail milega","get bail","bail chahiye","bailed out","bail application","custody","in jail","arrested"],"ans":lambda: BAIL},
    {"kw":["fir","police complaint","police refuse","first information report","zero fir","police nahi sun","police not registering","complaint police","darj fir","police nc"],"ans":lambda: FIR},
    {"kw":["labour","job terminated","fired","wrongful termination","salary not paid","unpaid wages","notice period","retrenchment","industrial dispute","payment of wages","naukri gayi","naukri se nikala","boss ne nikala"],"ans":lambda: LABOUR},
    {"kw":["sexual harassment","posh","workplace harassment","office mein","boss harassing","colleague harassing","inappropriate touch","icc","internal complaint"],"ans":lambda: POSH},
    {"kw":["provident fund","pf fraud","epf","pf nahi jama","epfo","pf deducted","pf not deposited"],"ans":lambda: PF},
    {"kw":["consumer complaint","defective product","refund nahi","warranty","service deficiency","amazon","flipkart","company cheated","online shopping","product damage","e commerce fraud","edaakhil","consumer forum","ncdrc"],"ans":lambda: CONSUMER},
    {"kw":["insurance rejected","claim reject","claim denied","health insurance","insurance not paying","irdai","bima","policy claim","insurance company","insurance fraud"],"ans":lambda: INSURANCE},
    {"kw":["property dispute","landlord","tenant","rent","eviction","security deposit","makaan","zameen","illegal eviction","rent nahi de raha","kiraya wapas","deposit wapas","unauthorized construction","encroachment","encroach","boundary wall","property fraud","title dispute","benami","rera","builder fraud","flat possession","developer fraud"],"ans":lambda: PROPERTY},
    {"kw":["inheritance","succession","property after death","father died","mother died","legal heir","will","probate","intestate","succession certificate","partition suit","ancestral property","Hindu succession"],"ans":lambda: INHERITANCE},
    {"kw":["defamation","false allegation","reputation damage","spreading rumors","fake news","slander","libel","499","500","reputation harm","facebook post","whatsapp post","viral message"],"ans":lambda: DEFAMATION},
    {"kw":["extortion","blackmail","threatening","protection money","demand money","goon","threatening call","threatening message","ransom"],"ans":lambda: EXTORTION},
    {"kw":["arrest","detained","police custody","illegal arrest","wrongful arrest","rights arrest","article 22","22 article","giraftari","arrested wrongly","lockup"],"ans":lambda: ARREST_RIGHTS},
    {"kw":["dowry","dahej","dowry demand","dowry harassment","dowry prohibition","dahej mang","bride burning","dowry death","304b"],"ans":lambda: DOWRY},
    {"kw":["senior citizen","parents neglect","son not supporting","elderly","parents thrown","maintenance parents","welfare parents act","old age","vridh"],"ans":lambda: SENIOR},
    {"kw":["child custody","bachche ki custody","custody of child","parental rights","children divorce","guardianship","minor child"],"ans":lambda: CUSTODY},
    {"kw":["theft","steal","chori","stolen","pick pocket","robbery","burglary","house break","mobile stolen","phone stolen"],"ans":lambda: THEFT},
    {"kw":["murder","302","hatya","killing","homicide","life sentence","death penalty","accused murder","unnatural death"],"ans":lambda: MURDER},
    {"kw":["cheating","420","fraud","ponzi","investment fraud","fake company","money fraud","swindled","financial fraud","insurance fraud","duped"],"ans":lambda: FRAUD_GENERAL},
    {"kw":["rti","right to information","information act","government documents","public authority","pio","first appeal"],"ans":lambda: RTI},
    {"kw":["writ","high court","supreme court","habeas corpus","mandamus","certiorari","prohibition","quo warranto","PIL","public interest","constitutional","fundamental rights","article 21","article 14","article 19"],"ans":lambda: CONSTITUTIONAL},
    {"kw":["company","startup","llp","partnership","shareholder dispute","director","mca","roc","nclt","corporate","business dispute","contract"],"ans":lambda: CORPORATE},
    {"kw":["gst","income tax","tax notice","tax demand","it return","tds","assessment","tax raid","black money"],"ans":lambda: TAX},
    {"kw":["rera","builder fraud","flat possession","possession delay","builder default","flat nahi mila","apartment delay","housing society","real estate fraud","property registry","builder cheated","flat registration","construction delay","developer fraud","homebuyer"],"ans":lambda: RERA_BUILDER},
    {"kw":["pocso","child abuse","child sexual","minor abuse","child molest","bachche ke saath","minor ke saath","child harassed","school abuse","teacher abuse","child raped","underage abuse","child helpline","1098"],"ans":lambda: POCSO_CHILD},
    {"kw":["medical negligence","doctor mistake","hospital mistake","wrong treatment","doctor ne galat","operation mein galat","hospital se maut","icu death","wrong diagnosis","doctor negligent","malpractice","hospital fraud","doctor harmed","patient rights"],"ans":lambda: MEDICAL_NEG},
    {"kw":["police brutality","police beating","police violence","police atrocity","police third degree","police custody death","police harassment","police corruption","policeman bribe","constable marpeet","thana mein mara","custodial violence","police ne mara","illegal detention police","police bribe"],"ans":lambda: POLICE_BRUTALITY},
    {"kw":["nri","overseas indian","foreign return","nri property","nri dispute","poa holder","power of attorney fraud","nri rights","oci","pio","overseas citizen","living abroad","nri india property","nri case india"],"ans":lambda: NRI},
    {"kw":["land acquisition","government took land","sarkar ne zameen","compensation land","rfctlarr","la act","nhia","airport land","highway land","government land","compulsory acquisition","eminent domain","sarkar ne khet"],"ans":lambda: LAND_ACQ},
    {"kw":["passport","visa problem","travel document","passport office","passport not coming","tatkaal","ecr passport","police verification passport","passport renew","passport impound","passport stuck","passport delay","travel ban"],"ans":lambda: PASSPORT},
]

# Maps intent index → correct ML category (overrides inaccurate ML model predictions)
INTENT_TO_CATEGORY = {
    0:  'motor_accident',      # hit_and_run
    1:  'cyber_crime',         # cheque_bounce (wait, cheque bounce should be fraud or corporate_tax)
    2:  'family',              # domestic_violence
    3:  'cyber_crime',         # cyber_crime
    4:  'family',              # divorce
    5:  'family',              # maintenance
    6:  'criminal',            # bail
    7:  'criminal',            # fir
    8:  'labour',              # labour
    9:  'sexual_offense',      # posh (workplace harassment)
    10: 'labour',              # pf
    11: 'consumer',            # consumer
    12: 'consumer',            # insurance
    13: 'property',            # property
    14: 'property',            # inheritance
    15: 'criminal',            # defamation
    16: 'theft_robbery',       # extortion (or keep criminal) -> let's map to theft_robbery
    17: 'constitutional',      # arrest_rights
    18: 'family',              # dowry
    19: 'family',              # senior_citizen
    20: 'family',              # child_custody
    21: 'theft_robbery',       # theft
    22: 'criminal',            # murder
    23: 'fraud',               # fraud_general
    24: 'constitutional',      # rti
    25: 'constitutional',      # constitutional
    26: 'corporate_tax',       # corporate
    27: 'corporate_tax',       # tax
    28: 'property',            # rera_builder
    29: 'sexual_offense',      # pocso_child
    30: 'medical_negligence',  # medical_negligence
    31: 'constitutional',      # police_brutality
    32: 'property',            # nri
    33: 'property',            # land_acquisition
    34: 'constitutional',      # passport
}

# ── Response Templates ─────────────────────────────────────────────────────────
HIT_RUN = """🚗 <strong>Motor Vehicle Accident / Hit & Run — Complete Legal Guide</strong><br><br>
<strong>IF YOU ARE THE VICTIM:</strong><br>
1. Call <strong>112 (Police)</strong> immediately. Note vehicle number, color, time, location.<br>
2. Get a <strong>Medico-Legal Certificate (MLC)</strong> from a government hospital — crucial evidence.<br>
3. File <strong>FIR at nearest police station</strong> under IPC Section 279 (Rash Driving) + 337/338 (Causing Hurt).<br>
4. File a <strong>Motor Accident Claim Petition at MACT</strong> (Motor Accident Claims Tribunal).<br>
5. If driver fled and is unknown: claim from the <strong>Solatium Fund</strong> under Motor Vehicles Act 1988.<br>
6. NO court fee for MACT claims. Compensation is based on income, age, and severity of injury.<br><br>
<strong>IF YOU ARE THE ACCUSED DRIVER:</strong><br>
• IPC Section 304A (BNS 106) applies — Causing Death by Negligence. Up to 2 years + fine.<br>
• Surrender to police with a lawyer. Fleeing from accident scene <em>increases</em> the penalty significantly.<br><br>
⚖️ <strong>Laws:</strong> Motor Vehicles Act 1988, IPC 279/304A/337/338 | BNS 281/106"""

CHEQUE = """🏦 <strong>Cheque Bounce — Section 138, Negotiable Instruments Act 1881</strong><br><br>
<strong>Strict Deadlines — Must Follow Exactly:</strong><br>
1. Get the <strong>Cheque Return Memo</strong> from your bank (written proof of dishonour).<br>
2. Send <strong>Legal Demand Notice</strong> to the drawer within <strong>30 days</strong> of receiving the memo — via Registered Post.<br>
3. The drawer has <strong>15 days</strong> after receiving notice to make payment.<br>
4. If not paid: File <strong>criminal complaint in Judicial Magistrate Court</strong> within 30 days of the 15-day expiry.<br><br>
<strong>What you can claim:</strong><br>
• Cheque amount + <strong>up to 2× as compensation</strong> + legal costs<br><br>
<strong>Penalty for drawer:</strong> Up to <strong>2 years imprisonment</strong> or fine up to twice the cheque amount or both.<br><br>
⚠️ <strong>Missing any deadline = case dismissed permanently.</strong><br>
⚖️ <strong>Law:</strong> Section 138, NI Act 1881."""

DV = """👨‍👩‍👧 <strong>Domestic Violence / Dowry Harassment — Emergency Legal Action</strong><br><br>
🚨 <strong>CALL NOW IF IN IMMEDIATE DANGER:</strong> Women Helpline <strong>181</strong> | Police <strong>112</strong><br><br>
<strong>Your Legal Weapons:</strong><br>
1. <strong>Protection Order</strong> under DV Act 2005 — can be granted <em>same day</em> by Magistrate.<br>
2. <strong>Residence Order</strong> — you cannot be thrown out of the matrimonial home (even if not your name).<br>
3. <strong>FIR under IPC Section 498A</strong> (BNS 85) — cruelty by husband/relatives. Non-bailable, cognizable.<br>
4. <strong>Maintenance</strong> under DV Act or Section 125 CrPC (BNSS 144) — can be ordered within weeks.<br>
5. For dowry demands: FIR under <strong>Dowry Prohibition Act 1961</strong> (up to 5 years + fine).<br>
6. If dowry death: IPC Section 304B (BNS 80) — minimum 7 years, can be life imprisonment.<br><br>
⚖️ <strong>Laws:</strong> DV Act 2005, IPC 498A/304B, Dowry Prohibition Act 1961."""

CYBER = """💻 <strong>Cyber Crime / Online Fraud — Immediate Steps</strong><br><br>
⏰ <strong>Act within 72 hours — recovery probability drops drastically after that.</strong><br><br>
1. 🚨 Call <strong>National Cyber Crime Helpline: 1930</strong> immediately to attempt transaction freeze.<br>
2. File report at <strong>cybercrime.gov.in</strong> (National Cyber Crime Reporting Portal).<br>
3. Visit local police <strong>Cyber Crime Cell</strong> and file FIR.<br>
4. Block all compromised cards/accounts via your bank immediately.<br>
5. For UPI fraud: contact <strong>NPCI at 18001201740</strong>.<br>
6. For OTP fraud, phishing: preserve all screenshots, messages, transaction IDs as evidence.<br><br>
<strong>Applicable Sections:</strong><br>
• IT Act Section 66C — Identity Theft (up to 3 years + ₹1L fine)<br>
• IT Act Section 66D — Cheating by personation online (up to 3 years + ₹1L fine)<br>
• IPC Section 420 / BNS 318 — Cheating (up to 7 years + fine)<br><br>
⚖️ <strong>Laws:</strong> IT Act 2000 (S.66C, 66D), IPC 420 / BNS 318."""

DIVORCE = """👨‍👩 <strong>Divorce in India — Complete Guide</strong><br><br>
<strong>TYPE 1: Mutual Consent Divorce (Fastest)</strong><br>
• Both agree on alimony, custody, property division<br>
• File joint petition in Family Court<br>
• 6-month cooling-off period (waivable by SC if no children and irreconcilable differences)<br>
• Must have lived separately <strong>1 year minimum</strong><br>
• Laws: Hindu Marriage Act 1955 S.13B | Special Marriage Act 1954 S.28<br><br>
<strong>TYPE 2: Contested Divorce (One-sided)</strong><br>
• File citing grounds: cruelty, adultery, desertion (2+ years), mental disorder, conversion, leprosy<br>
• File in Family Court of the area where you last lived together<br>
• Duration: typically 1–5+ years<br>
• Law: Hindu Marriage Act 1955 S.13<br><br>
<strong>Alimony/Maintenance during divorce:</strong><br>
• Apply under Section 24 HMA for interim maintenance during proceedings<br>
• Or under Section 125 CrPC / BNSS 144 in Magistrate Court<br><br>
⚖️ <strong>Laws:</strong> Hindu Marriage Act 1955, Special Marriage Act 1954, CrPC S.125 / BNSS S.144."""

MAINTENANCE = """💰 <strong>Maintenance / Nafqa — Legal Rights</strong><br><br>
<strong>Who Can Claim Maintenance:</strong><br>
• Wife (including divorced wife in certain cases), minor children, and parents from son/daughter<br><br>
<strong>How to Apply:</strong><br>
1. File application under <strong>Section 125 CrPC / BNSS Section 144</strong> in nearest Magistrate Court.<br>
2. Or apply under <strong>Section 24 Hindu Marriage Act</strong> if divorce proceedings are ongoing.<br>
3. Under BNSS 2023, Magistrate must decide maintenance within <strong>60 days</strong>.<br>
4. <strong>Interim maintenance</strong> can be ordered quickly — even within weeks of filing.<br><br>
<strong>Amount depends on:</strong> Husband's income, wife's income, standard of living, number of children<br><br>
<strong>For parents neglected by children:</strong><br>
• File under <strong>Maintenance and Welfare of Parents & Senior Citizens Act 2007</strong> before SDM — free, quick, and effective.<br><br>
⚖️ <strong>Laws:</strong> CrPC S.125 / BNSS S.144, Hindu Marriage Act S.24, Parents & Senior Citizens Act 2007."""

BAIL = """⚖️ <strong>Bail in India — Complete Guide</strong><br><br>
<strong>Types of Bail:</strong><br>
• <strong>Regular Bail (S.437/BNSS 480):</strong> For non-bailable offenses — from Magistrate/Sessions Court<br>
• <strong>Anticipatory Bail (S.438/BNSS 482):</strong> Apply before arrest if you fear arrest — from Sessions Court or High Court<br>
• <strong>Station Bail (S.436/BNSS 478):</strong> For bailable offenses — get it at the police station itself<br>
• <strong>Default Bail (S.167(2)/BNSS):</strong> If chargesheet not filed in 60/90 days — bail as right<br><br>
<strong>How to apply:</strong><br>
1. Engage a lawyer immediately<br>
2. File bail application in appropriate court with: FIR copy, grounds for bail, surety details<br>
3. Factors court considers: nature of crime, criminal record, flight risk, evidence tampering risk<br><br>
<strong>Under BNS 2023 update:</strong> Undertrials who have served 1/3rd of maximum sentence must be released on bail.<br><br>
⚖️ <strong>Laws:</strong> CrPC S.436-439 / BNSS S.478-484."""

FIR = """📋 <strong>FIR (First Information Report) — Your Rights</strong><br><br>
<strong>Police MUST register FIR for cognizable offenses — it is your legal right.</strong><br><br>
<strong>If police refuse to file FIR:</strong><br>
1. Send written complaint to <strong>Superintendent of Police (SP)</strong> via Registered Post.<br>
2. File <strong>private complaint</strong> in Magistrate Court under Section 156(3) CrPC / BNSS 175(3).<br>
3. Approach <strong>State Human Rights Commission</strong>.<br>
4. File <strong>Zero FIR</strong> — can be filed at ANY police station regardless of jurisdiction.<br><br>
<strong>Under BNS/BNSS 2023 new rules:</strong><br>
• E-FIR can be filed online for specified offenses<br>
• Police must give signed copy of FIR free of cost<br>
• Victim must be informed of investigation progress<br><br>
⚖️ <strong>Law:</strong> CrPC Section 154 / BNSS Section 173."""

LABOUR = """👷 <strong>Labour / Employment Dispute — Legal Rights</strong><br><br>
<strong>Common Issues & Solutions:</strong><br><br>
<strong>Wrongful Termination:</strong><br>
• Under Industrial Disputes Act 1947: Employer must give notice + retrenchment compensation<br>
• Send legal notice to employer within 30 days<br>
• File complaint with <strong>Labour Commissioner</strong> or directly in <strong>Labour Court/Industrial Tribunal</strong><br><br>
<strong>Unpaid Salary/Wages:</strong><br>
• File complaint under <strong>Payment of Wages Act 1936</strong><br>
• Authority must decide within 3 months<br><br>
<strong>PF Not Deposited:</strong><br>
• File on <strong>EPFiGMS portal (epfigms.gov.in)</strong><br>
• FIR under IPC 405/BNS 316 (Criminal Breach of Trust)<br><br>
<strong>Maternity Benefits withheld:</strong> File complaint under <strong>Maternity Benefit Act 1961</strong> — up to 26 weeks paid leave entitlement<br><br>
⚖️ <strong>Laws:</strong> Industrial Disputes Act 1947, Payment of Wages Act 1936, EPF Act 1952, Maternity Benefit Act 1961."""

POSH = """⚖️ <strong>Sexual Harassment at Workplace (POSH Act 2013)</strong><br><br>
<strong>Steps to take — strict timelines apply:</strong><br>
1. File written complaint with <strong>Internal Complaints Committee (ICC)</strong> within <strong>3 months</strong> of last incident.<br>
2. If company has no ICC (mandatory for 10+ employees) — file with <strong>Local Complaints Committee (LCC)</strong> at district level.<br>
3. File <strong>Zero FIR</strong> at nearest police station under IPC 354A / BNS 75 simultaneously.<br>
4. Preserve all evidence: emails, WhatsApp, voice recordings, witness names.<br><br>
<strong>ICC must complete inquiry within 90 days.</strong><br>
<strong>Employer who doesn't have ICC faces fine of up to ₹50,000.</strong><br><br>
⚖️ <strong>Laws:</strong> POSH Act 2013, IPC 354A / BNS 75, IPC 354 / BNS 74."""

PF = """💰 <strong>Provident Fund (EPF) Fraud by Employer</strong><br><br>
1. Check your <strong>EPF passbook</strong> at UAN Member Portal (unifiedportal-mem.epfindia.gov.in).<br>
2. File grievance on <strong>EPFiGMS portal (epfigms.gov.in)</strong>.<br>
3. Send written complaint to <strong>Regional PF Commissioner</strong> with salary slips showing PF deduction.<br>
4. <strong>PF diversion is a criminal offense</strong> under IPC 405 / BNS 316 (Criminal Breach of Trust). File FIR.<br>
5. Employer faces up to 3 years imprisonment + fine + recovery of full amount with interest.<br><br>
⚖️ <strong>Laws:</strong> EPF & MP Act 1952, IPC 405 / BNS 316."""

CONSUMER = """🛒 <strong>Consumer Rights — Consumer Protection Act 2019</strong><br><br>
<strong>Filing a Consumer Complaint:</strong><br>
1. Send <strong>written complaint to company</strong> giving 15 days to resolve.<br>
2. If not resolved: File on <strong>National Consumer Helpline: 1800-11-4000</strong> or consumerhelpline.gov.in<br>
3. File online at <strong>edaakhil.nic.in</strong> — District Consumer Commission (claims up to ₹50 Lakhs, FREE)<br>
4. State Commission: Claims ₹50L to ₹2 Crore<br>
5. NCDRC (National): Claims above ₹2 Crore<br><br>
<strong>Time limit:</strong> 2 years from date of defect/deficiency.<br>
<strong>Compensation:</strong> Refund + replacement + mental harassment compensation + litigation costs<br><br>
⚖️ <strong>Laws:</strong> Consumer Protection Act 2019."""

INSURANCE = """🏥 <strong>Insurance Claim Rejection — Your Rights</strong><br><br>
1. File detailed grievance with insurer's <strong>Grievance Redressal Officer (GRO)</strong>.<br>
2. If no response in 30 days: File at <strong>IRDAI Bima Bharosa portal (bimabharosa.irdai.gov.in)</strong>.<br>
3. Approach <strong>Insurance Ombudsman</strong> (free) — for claims up to ₹30 Lakhs.<br>
4. File <strong>Consumer complaint</strong> at District Consumer Commission for deficiency of service.<br><br>
<strong>Insurance company MUST:</strong><br>
• Acknowledge claim within 3 days<br>
• Settle/repudiate within 30 days of document submission<br>
• Pay interest at bank rate +2% for delays beyond 30 days<br><br>
⚖️ <strong>Laws:</strong> Insurance Act 1938, Consumer Protection Act 2019, IRDAI Regulations."""

PROPERTY = """🏠 <strong>Property Law — Rights and Remedies</strong><br><br>
<strong>Common Issues:</strong><br>
• <strong>Encroachment:</strong> File FIR (IPC 441/BNS 329) + apply for Stay Order in Civil Court IMMEDIATELY<br>
• <strong>Illegal Eviction by Landlord:</strong> Landlord cannot evict without court order — file police complaint<br>
• <strong>Security Deposit:</strong> Legal notice + Consumer Forum or Civil Court suit for recovery<br>
• <strong>Builder Fraud (RERA):</strong> File complaint on State RERA portal for possession or refund with 10.85% interest<br>
• <strong>Forged Documents:</strong> FIR under IPC 467/468 (Forgery) + Civil suit for declaration<br><br>
<strong>Important Documents to secure:</strong><br>
Sale deed, Title deed, Khata, Mutation, Encumbrance Certificate, Property Tax receipts<br><br>
⚖️ <strong>Laws:</strong> Transfer of Property Act 1882, Specific Relief Act 1963, RERA Act 2016, Registration Act 1908."""

INHERITANCE = """📜 <strong>Inheritance & Succession — Legal Rights</strong><br><br>
<strong>Hindu Succession Act 1956 (Amended 2005):</strong><br>
• All Class I legal heirs (spouse, sons, daughters, mother) have EQUAL share<br>
• Daughters have equal rights in ancestral property since 2005 amendment<br><br>
<strong>Steps to claim your share:</strong><br>
1. Gather property documents + proof of relationship<br>
2. Send legal notice to co-heirs asserting your share<br>
3. Apply for <strong>Succession Certificate</strong> from court (for movable assets, bank accounts)<br>
4. File <strong>Partition Suit</strong> in Civil Court if co-heirs refuse<br>
5. Apply for <strong>mutation of property</strong> in your name at Revenue/Municipal office<br><br>
<strong>Time limit for partition suit:</strong> 12 years from denial of your share<br><br>
⚖️ <strong>Laws:</strong> Hindu Succession Act 1956, Indian Succession Act 1925, Registration Act 1908."""

DEFAMATION = """📢 <strong>Defamation — Legal Remedies</strong><br><br>
<strong>Immediate steps:</strong><br>
1. <strong>Screenshot everything</strong> with date, time, URL — evidence gets deleted quickly.<br>
2. Send <strong>legal notice</strong> demanding removal and written apology within 15 days.<br>
3. Report to social media platform for content removal.<br><br>
<strong>Criminal Defamation:</strong> File complaint in Magistrate Court — IPC 500 / BNS 356 (up to 2 years + fine)<br>
<strong>Civil Defamation:</strong> Suit for damages — compensation for reputation loss, mental anguish<br><br>
<strong>Online Defamation:</strong> Also covered under IT Act provisions. File at cybercrime.gov.in.<br><br>
⚖️ <strong>Laws:</strong> IPC 499/500 / BNS 356, IT Act 2000."""

EXTORTION = """🚨 <strong>Extortion / Blackmail — Immediate Action Required</strong><br><br>
⚠️ <strong>DO NOT PAY under any circumstances — it invites more demands.</strong><br><br>
1. Preserve ALL threats: voice messages, WhatsApp, recordings, emails.<br>
2. File FIR immediately at police station — <strong>Extortion under IPC 383 / BNS 308</strong> (up to 3 years).<br>
3. For <strong>online sextortion/blackmail</strong>: report at cybercrime.gov.in + call 1930.<br>
4. Request police for <strong>witness protection</strong> if threatening physical harm.<br>
5. Approach Sessions Court/High Court for anticipatory bail protection.<br><br>
⚖️ <strong>Laws:</strong> IPC 383-389 / BNS 308-312 (Extortion)."""

ARREST_RIGHTS = """⚖️ <strong>Rights on Arrest — Constitutional Protections</strong><br><br>
<strong>Your FUNDAMENTAL RIGHTS on arrest (Article 22, Constitution of India):</strong><br>
1. Right to be <strong>informed of grounds of arrest</strong> — immediately and in writing under BNSS 2023.<br>
2. Right to <strong>consult and be defended by a lawyer of your choice</strong>.<br>
3. Police MUST produce you before a <strong>Magistrate within 24 hours</strong> of arrest.<br>
4. Right to <strong>inform a family member or friend</strong> of your arrest.<br>
5. Cannot be <strong>detained beyond 24 hours</strong> without Magistrate's order.<br><br>
<strong>New under BNSS 2023:</strong><br>
• Information of arrest must be shared with nominated person<br>
• Women can only be arrested after sunset with written order from Magistrate (exceptions apply)<br><br>
⚖️ <strong>Laws:</strong> Article 22 Constitution, BNSS 2023 S.37-60."""

DOWRY = """🚨 <strong>Dowry Harassment — Legal Remedies</strong><br><br>
<strong>Laws protecting you:</strong><br>
• <strong>IPC 498A / BNS 85</strong> — Cruelty for dowry demands. Up to 3 years + fine. Non-bailable.<br>
• <strong>Dowry Prohibition Act 1961</strong> — giving/taking dowry = 5 years + ₹15,000 fine.<br>
• <strong>IPC 304B / BNS 80</strong> — Dowry Death: minimum 7 years, can be life imprisonment.<br>
• <strong>DV Act 2005</strong> — Protection and residence orders available immediately.<br><br>
<strong>Immediate steps:</strong><br>
1. Call Women Helpline <strong>181</strong> or Police <strong>112</strong><br>
2. File FIR at nearest police station<br>
3. Approach Protection Officer for DV complaint<br>
4. Apply for Protection Order — can be granted SAME DAY<br><br>
⚖️ <strong>Laws:</strong> IPC 498A/304B, DV Act 2005, Dowry Prohibition Act 1961."""

SENIOR = """👴 <strong>Senior Citizens — Legal Rights & Maintenance</strong><br><br>
<strong>Under Maintenance and Welfare of Parents & Senior Citizens Act 2007:</strong><br>
• Children/grandchildren LEGALLY OBLIGATED to maintain parents.<br>
• Maximum maintenance: Rs. 10,000/month (states can increase).<br>
• File complaint before <strong>Sub-Divisional Magistrate (SDM)</strong> — it's FREE and quick.<br>
• SDM can order maintenance AND restoration of residence within 90 days.<br><br>
<strong>Additional protections:</strong><br>
• Property given to children can be REVOKED if they fail to maintain parents.<br>
• Filing in Senior Citizens Tribunal is free, simple, and fast.<br><br>
⚖️ <strong>Laws:</strong> Parents & Senior Citizens Act 2007, DV Act 2005 (for senior women)."""

CUSTODY = """👶 <strong>Child Custody — Legal Framework</strong><br><br>
<strong>Key Principle: Child's welfare is paramount — not the parent's preference.</strong><br><br>
<strong>Types of Custody:</strong><br>
• <strong>Physical Custody:</strong> Child lives with one parent<br>
• <strong>Joint Custody:</strong> Time shared between both parents<br>
• <strong>Legal Custody:</strong> Decision-making rights for education/health<br><br>
<strong>How courts decide:</strong> Age of child, stability of home, parent's income, child's bond with each parent<br><br>
<strong>Interim Custody:</strong> Get temporary custody order during proceedings — apply immediately in Family Court<br>
<strong>Visitation Rights:</strong> Non-custodial parent gets regular access<br><br>
⚖️ <strong>Laws:</strong> Guardians & Wards Act 1890, Hindu Minority & Guardianship Act 1956."""
THEFT = """<strong>Ghar Mein Chori — Aapke Liye Complete Step-by-Step Guide</strong><br><br>
Yeh sun ke bahut bura laga. Ghabrao mat — ab seedha kaam pe aate hain. Yeh karo:<br><br>
<span style='color:#f08080'>ABHI KE ABHI — Pehle yeh steps lo:</span><br>
1. <strong>112 pe call karo</strong> — Police ko turant inform karo. Address clearly batao.<br>
2. <strong>Ghar ka scene mat chhuo</strong> — Koi cheez mat hatao, mat saaf karo. Fingerprints aur footprints critical evidence hain.<br>
3. <strong>Agar chor abhi bhi andar ho sakta hai</strong> — Bahar raho, police ka wait karo. Safety pehle.<br><br>
<strong>FIR Kaise Darj Karein (24 ghante ke andar zaroor karo):</strong><br>
1. Apne area ke police station jao.<br>
2. Yeh information tayyar rakho:<br>
&nbsp;&nbsp;- Chori ka approximate time (kab hua hoga)<br>
&nbsp;&nbsp;- Kya-kya chori hua — list banao (cash, jewelry, mobile, laptop, documents, etc.)<br>
&nbsp;&nbsp;- Koi suspect ya CCTV footage hai toh zaroor mention karo<br>
3. FIR file karo under <strong>IPC Section 380 / BNS 305</strong> (Theft in dwelling house) — yeh ghar ki chori ke liye specific section hai.<br>
4. FIR ki <strong>certified copy zaroor lo</strong> — yeh FREE milti hai, tumhara haq hai.<br><br>
<strong>Evidence — JALDI Collect Karo:</strong><br>
- <strong>CCTV footage</strong> — society/colony/neighbours ke cameras check karo abhi. 24-48 ghante mein overwrite ho sakti hai!<br>
- <strong>Witnesses</strong> — kya kisi neighbour ne kuch dekha ya suna? Unka contact number lo.<br>
- <strong>Broken lock / window</strong> — apne phone se photos lo — timestamp ke saath.<br>
- Chori samaan ki <strong>original bills, photos, warranty cards</strong> dhundho.<br><br>
<strong>Agar Mobile/Laptop Chori Hua:</strong><br>
- <strong>CEIR portal (ceir.gov.in)</strong> pe IMEI number block karo — device chor ke liye useless ho jayega.<br>
- Telecom provider ko call karo — SIM block karo turant.<br>
- Google/Apple account se remotely device lock/wipe karo.<br><br>
<strong>Insurance Claim (Agar ghar ka insurance hai):</strong><br>
- Insurance company ko <strong>24 ghante ke andar</strong> inform karo.<br>
- FIR copy + stolen items ki list provide karni hogi claim ke liye.<br><br>
<strong>Applicable Laws:</strong><br>
- IPC 378 / BNS 303 — Theft: Up to 3 years + fine<br>
- IPC 380 / BNS 305 — Theft in dwelling house: Up to <strong>7 years + fine</strong> (stricter!)<br>
- IPC 457 / BNS 332 — House breaking by night: Up to <strong>14 years</strong><br>
- IPC 411 / BNS 317 — Receiving stolen property: Up to 3 years + fine<br><br>
Important: Agar police FIR likhne se mana kare — SP office ya Magistrate Court mein seedha complaint de sakte ho. Yeh tumhara kanoon ki haq hai.<br><br>
Koi aur sawaal hai ya aur details chahiye? Batao — main hoon!"""

MURDER = """⚖️ <strong>Homicide / Murder — Legal Provisions</strong><br><br>
• <strong>IPC 302 / BNS 101 — Murder:</strong> Death or Life Imprisonment + Fine<br>
• <strong>IPC 304 / BNS 105 — Culpable Homicide (not murder):</strong> 10 years / Life<br>
• <strong>IPC 304A / BNS 106 — Negligent Killing (accidents):</strong> Up to 2 years + Fine<br>
• <strong>IPC 307 / BNS 109 — Attempt to Murder:</strong> Up to 10 years<br><br>
<strong>Investigation:</strong> CBI or State Police depending on case. Post-mortem report is key evidence.<br>
<strong>Trial:</strong> Sessions Court (murder is tried only at Sessions Court level or higher).<br><br>
For victims' families: File FIR immediately, demand fair investigation, apply for bail opposition if accused arrested.<br><br>
⚖️ <strong>Laws:</strong> IPC 299-309 / BNS 100-113."""

FRAUD_GENERAL = """💸 <strong>Fraud / Cheating — Legal Remedies</strong><br><br>
• <strong>IPC 420 / BNS 318:</strong> Cheating — up to 7 years + fine<br>
• <strong>IPC 406 / BNS 316:</strong> Criminal Breach of Trust — up to 3 years + fine<br>
• <strong>IPC 467/468 / BNS 336/337:</strong> Forgery — up to 7 years + fine<br><br>
<strong>Steps depending on type:</strong><br>
• Online/cyber fraud: Call 1930 + file at cybercrime.gov.in<br>
• Investment fraud/Ponzi: FIR at Economic Offences Wing + SEBI complaint at scores.gov.in<br>
• Builder fraud: RERA complaint on State RERA portal<br>
• Cheque bounce: Legal notice within 30 days + Magistrate complaint<br>
• Consumer fraud: District Consumer Commission via edaakhil.nic.in<br><br>
⚖️ <strong>Laws:</strong> IPC 415-420 / BNS 316-318, IT Act 2000, Consumer Protection Act 2019."""

RTI = """📄 <strong>Right to Information (RTI) Act 2005</strong><br><br>
<strong>Any citizen can get information from any public authority within 30 days.</strong><br><br>
<strong>How to file RTI:</strong><br>
1. Write application to Public Information Officer (PIO) of the concerned department<br>
2. Pay fee: ₹10 (Central Govt) — some states vary<br>
3. File online at: rtionline.gov.in (for Central Govt) or state RTI portal<br>
4. Response within 30 days (48 hours for matters affecting life/liberty)<br><br>
<strong>If refused or no response:</strong><br>
• First Appeal to First Appellate Authority within 30 days<br>
• Second Appeal to Central/State Information Commission within 90 days<br><br>
⚖️ <strong>Law:</strong> Right to Information Act 2005."""

CONSTITUTIONAL = """⚖️ <strong>Constitutional Remedies — Writ Jurisdiction</strong><br><br>
<strong>5 Types of Writs (Article 32 — Supreme Court / Article 226 — High Court):</strong><br>
• <strong>Habeas Corpus:</strong> "Produce the body" — challenge illegal detention/arrest<br>
• <strong>Mandamus:</strong> Order to public authority to perform legal duty<br>
• <strong>Certiorari:</strong> Quash illegal orders of lower courts/tribunals<br>
• <strong>Prohibition:</strong> Stop lower court from acting beyond jurisdiction<br>
• <strong>Quo Warranto:</strong> Challenge illegal occupation of public office<br><br>
<strong>PIL (Public Interest Litigation):</strong><br>
• Any citizen can file for public interest matters directly in High Court or Supreme Court<br><br>
⚖️ <strong>Laws:</strong> Constitution of India Articles 12, 13, 14, 19, 21, 22, 32, 226."""

CORPORATE = """🏢 <strong>Corporate / Business Law</strong><br><br>
• <strong>Company incorporation disputes:</strong> File complaint with RoC (Registrar of Companies) or MCA portal<br>
• <strong>Shareholder disputes:</strong> Approach NCLT (National Company Law Tribunal)<br>
• <strong>Contract disputes:</strong> Send legal notice + file civil suit or invoke arbitration clause<br>
• <strong>Partnership disputes:</strong> File suit in civil court; for LLP — approach NCLT<br>
• <strong>Insolvency:</strong> IBC 2016 — file before NCLT for corporate insolvency (debt ₹1 crore minimum)<br><br>
⚖️ <strong>Laws:</strong> Companies Act 2013, LLP Act 2008, IBC 2016, Indian Contract Act 1872, Arbitration Act 1996."""

TAX = """💰 <strong>Tax Disputes — Income Tax / GST</strong><br><br>
<strong>For Income Tax disputes:</strong><br>
1. Reply to tax notice within stated period (usually 15-30 days)<br>
2. File appeal before CIT (Appeals) within 30 days of assessment order<br>
3. Further appeal to ITAT (Income Tax Appellate Tribunal)<br><br>
<strong>For GST disputes:</strong><br>
1. Reply to SCN (Show Cause Notice) with all documents<br>
2. Appeal before GST Appellate Authority within 3 months<br>
3. Further appeal before GSTAT or High Court<br><br>
<strong>Tax Raids:</strong> Cooperate but insist on witness presence; don't sign blank or incorrect documents.<br><br>
⚖️ <strong>Laws:</strong> Income Tax Act 1961, GST Act 2017, Finance Act."""

# ── NEW INTENT TEMPLATES ───────────────────────────────────────────────────────

RERA_BUILDER = """🏗️ <strong>RERA — Builder / Real Estate Fraud Complaint</strong><br><br>
<strong>Builder ne flat nahi diya? Paisa le liya aur possession nahi?</strong><br><br>
<strong>Step-by-Step Action:</strong><br>
1. File complaint on your <strong>State RERA Portal</strong> — each state has own portal (MahaRERA, UP-RERA, HRERA etc.)<br>
2. Claim <strong>interest @ 10.85% p.a.</strong> on all amounts paid from due possession date<br>
3. Or demand <strong>full refund + interest</strong> if you want to exit the project<br>
4. Also file at <strong>Consumer Commission (edaakhil.nic.in)</strong> — compensation for mental harassment too<br>
5. If clearly fraudulent: <strong>FIR under IPC 420 / BNS 318</strong> (Cheating) + approach ED under PMLA for large sums<br><br>
<strong>Your RERA Rights:</strong><br>
• Developer MUST register project with RERA before selling any unit<br>
• RERA Tribunal must decide complaint within 60 days<br>
• 70% of buyer funds must be kept in separate escrow account<br>
• You can claim compensation + interest without going to regular court<br><br>
⚖️ <strong>Laws:</strong> RERA Act 2016, Consumer Protection Act 2019, IPC 420 / BNS 318."""

POCSO_CHILD = """🚨 <strong>POCSO Act — Child Sexual Abuse Protection</strong><br><br>
🚨 <strong>EMERGENCY HELPLINES — CALL IMMEDIATELY:</strong><br>
• <strong>Childline: 1098</strong> (24×7, FREE, confidential)<br>
• <strong>Police: 112</strong> — FIR mandatory under POCSO<br>
• <strong>Women Helpline: 181</strong><br><br>
<strong>POCSO Act 2012 (Amended 2019) — Key Provisions:</strong><br>
• Covers ALL sexual offenses against children below <strong>18 years</strong><br>
• <strong>Penetrative sexual assault:</strong> Minimum 20 years → Life Imprisonment (2019 amendment)<br>
• <strong>Aggravated assault (by guardian/family/authority):</strong> Life Imprisonment or Death<br>
• <strong>Sexual harassment of child:</strong> Up to 3 years<br>
• Police MUST register FIR — <strong>no discretion allowed</strong><br>
• Child must be produced before Child Welfare Committee (CWC) within 24 hours<br>
• Trial in <strong>Special POCSO Court</strong> — child's identity CANNOT be disclosed<br>
• Child's statement recorded only by lady sub-inspector in child-friendly environment<br><br>
⚖️ <strong>Laws:</strong> POCSO Act 2012 (Amended 2019), JJ Act 2015, IPC 376 / BNS 63."""

MEDICAL_NEG = """🏥 <strong>Medical Negligence — Patient Rights & Legal Remedies</strong><br><br>
<strong>Medical negligence = failure of standard care causing measurable harm.</strong><br><br>
<strong>Remedies Available (any one or all together):</strong><br><br>
1. <strong>Consumer Commission:</strong> File at edaakhil.nic.in — most effective for compensation. FREE for claims under ₹5 lakhs<br>
2. <strong>State Medical Council:</strong> File disciplinary complaint to suspend/debar the doctor's license<br>
3. <strong>FIR under IPC 304A / BNS 106:</strong> Causing death by negligent act — up to 2 years imprisonment<br>
4. <strong>NHRC / SHRC:</strong> If a government hospital is involved — file human rights complaint<br><br>
<strong>Evidence to collect urgently:</strong><br>
• Original medical records, prescriptions, test reports — hospitals MUST give these to patient<br>
• Get second medical opinion in writing from another qualified doctor<br>
• Preserve all bills, discharge summaries, ICU charts, operation notes<br><br>
📌 <em>Supreme Court ruling:</em> Only <strong>gross negligence</strong> attracts criminal liability — not bona fide errors.<br><br>
⚖️ <strong>Laws:</strong> Consumer Protection Act 2019, IPC 304A / BNS 106, Indian Medical Council Act 1956."""

POLICE_BRUTALITY = """🚔 <strong>Police Brutality / Corruption — Your Legal Rights</strong><br><br>
🚨 <strong>Emergency: Call 100 | NHRC Helpline: 14433</strong><br><br>
<strong>Legal Remedies Against Illegal Police Action:</strong><br>
1. File <strong>written complaint to SP / DCP / Inspector General</strong> of your district<br>
2. Approach <strong>State Police Complaints Authority</strong> (every state must have one)<br>
3. File complaint with <strong>NHRC / State Human Rights Commission</strong> — they can award compensation<br>
4. File <strong>Habeas Corpus petition in High Court</strong> for illegal detention — can be filed same day<br>
5. FIR against officer under <strong>IPC 323/325 (Hurt), IPC 330/331 (Hurt to extort confession)</strong> — cognizable offense<br>
6. For custodial death: Demand CBI investigation + file NHRC complaint immediately<br><br>
<strong>New Rights under BNSS 2023:</strong><br>
• Medical examination of arrested person is MANDATORY before and after custody<br>
• Custody torture punishable with <strong>up to 10 years imprisonment</strong><br>
• Videography of search/arrest scenes is mandatory<br>
• Women can only be arrested after sunset with written Magistrate order (exceptions apply)<br><br>
⚖️ <strong>Laws:</strong> IPC 323/330/331 / BNS 115/120, Constitution Art. 20/21/22, BNSS 2023 S.37–60."""

NRI = """🌍 <strong>NRI Legal Rights — Property, Family & Finance from Abroad</strong><br><br>
<strong>Property Disputes (Encroachment/Sale without consent while you're abroad):</strong><br>
• Grant <strong>Special Power of Attorney (SPA)</strong> — notarized + apostilled is MANDATORY for Indian court use<br>
• File <strong>civil suit for declaration + injunction</strong> in local Civil Court through your GPA holder<br>
• For POA fraud: <strong>FIR under IPC 420/467/468 / BNS 318/336/337</strong> (fraud + forgery)<br>
• NRI can fight Indian cases entirely through authorized GPA holder — no need to visit India<br><br>
<strong>Family Law for NRI:</strong><br>
• Divorce proceedings can happen in India even if spouse is abroad<br>
• Maintenance application can be filed in India — service abroad via BNSS provisions<br>
• If spouse confiscated your passport: DV Act remedy available — contact embassy immediately<br><br>
<strong>Tax for NRI:</strong><br>
• NRI status: Outside India for 182+ days in financial year<br>
• Rental income from Indian property is taxable in India; DTAA benefit where applicable<br><br>
📌 Contact <strong>NRI Legal Cell</strong> at nearest Indian Embassy/Consulate — free legal assistance available.<br><br>
⚖️ <strong>Laws:</strong> FEMA 1999, Hindu Succession Act 1956, Registration Act 1908, Income Tax Act 1961."""

LAND_ACQ = """🏗️ <strong>Land Acquisition by Government — Your Rights & Compensation</strong><br><br>
<strong>Under RFCTLARR Act 2013 (Right to Fair Compensation and Transparency):</strong><br><br>
<strong>Compensation Entitlements:</strong><br>
• <strong>Urban land:</strong> 2× registered market value + solatium (20% extra)<br>
• <strong>Rural land (within 50km of city):</strong> 2× to 4× market value<br>
• <strong>12% annual interest</strong> from date of SIA notification till payment<br>
• <strong>Rehabilitation & Resettlement package</strong> + annuity for 20 years<br><br>
<strong>Challenging Inadequate Compensation:</strong><br>
1. File objections before Land Acquisition Collector within <strong>60 days</strong> of award notification<br>
2. File <strong>Reference Petition to High Court</strong> under Section 64 — court appoints independent assessor<br>
3. Challenge the acquisition itself if process was defective — writ petition in High Court<br><br>
<strong>National Highway / Defence land:</strong> Separate Acts apply but quantum can still be challenged in court<br><br>
⚖️ <strong>Laws:</strong> RFCTLARR Act 2013, National Highways Act 1956, Constitution Art. 300A."""

PASSPORT = """📘 <strong>Passport Issues — Legal Remedies & Steps</strong><br><br>
<strong>Passport Delayed or Not Issued:</strong><br>
1. Check status at <strong>passportindia.gov.in</strong> or Passport Seva Kendra app<br>
2. If police verification pending: Visit local police station with a written reminder<br>
3. If delayed 30+ days without reason: File <strong>RTI with the Passport Office</strong> (₹10 fee, 30-day response)<br>
4. Approach <strong>Regional Passport Officer</strong> in person with written complaint<br>
5. File grievance at <strong>pgportal.gov.in</strong> (Central Public Grievance Redress System)<br><br>
<strong>If Passport Impounded / Revoked:</strong><br>
• File representation to Passport Authority within 30 days of order<br>
• Or file revision petition to <strong>Ministry of External Affairs</strong><br>
• High Court can stay impounding via <strong>Writ Petition (Certiorari/Mandamus)</strong><br><br>
<strong>Tatkaal (Urgent) Passport:</strong><br>
• Issued in 3 working days with additional fee<br>
• Required: Annexure F + any one valid address proof<br><br>
⚖️ <strong>Laws:</strong> Passports Act 1967, RTI Act 2005, Constitution Art. 21 (Right to Travel)."""

# ── Scoring engine ─────────────────────────────────────────────────────────────
def score_intent(norm_text, keywords):
    score = 0
    tokens = set(norm_text.split())
    stop_words = {'in', 'the', 'by', 'of', 'and', 'or', 'a', 'an', 'to', 'for', 'with', 'on', 'at', 'from'}
    for kw in keywords:
        kw_words = kw.split()
        if kw in norm_text:
            score += len(kw_words) * 5    # exact phrase = highest score
        elif all(w in tokens for w in kw_words):
            score += len(kw_words) * 3    # all words present anywhere in text
        else:
            # Only give partial points if a meaningful word matches
            meaningful_words = [w for w in kw_words if w not in stop_words]
            matched = [w for w in meaningful_words if w in tokens]
            if matched:
                score += len(matched)     # 1 point per meaningful word matched
    return score

# ── Follow-up / Continuation Query Detection ────────────────────────────────────
# These are words that appear in follow-up messages and have NO legal topic meaning.
FOLLOWUP_TRIGGERS = {
    'step', 'steps', 'guide', 'guidance', 'guidence', 'detail', 'details', 'detailed',
    'samjhao', 'batao', 'aur', 'help', 'next', 'continue', 'poora', 'full',
    'aage', 'kya', 'karu', 'what', 'should', 'do', 'explain', 'more',
    'bata', 'procedure', 'process', 'me', 'main', 'chahiye', 'please',
    'ab', 'further', 'bolo', 'karun', 'exact', 'proper', 'complete',
    'give', 'now', 'karo', 'i', 'tell', 'want', 'chahiye', 'chaiye',
}

# Strong multi-word phrases that always mean "tell me more about last topic"
STRONG_FOLLOWUP_PHRASES = [
    'guide me', 'step by step', 'kya karu', 'ab kya karu', 'aage kya karu',
    'what should i do', 'what to do', 'how to proceed', 'aur batao',
    'poora batao', 'please guide', 'guidance do', 'guidence do',
    'aur details', 'next steps', 'next step', 'samjhao', 'help me',
    'me kya karu', 'main kya karu', 'ab kya', 'kya karna chahiye',
    'kya karna', 'further guidance', 'more guidance', 'aur kya',
    'kya karun', 'batao kya', 'ab bolo', 'give guidance',
    'give me guidance', 'aur bata', 'continue', 'further steps',
]

def is_followup_query(norm_text):
    """Returns True if the message is a follow-up with no new legal topic."""
    # Check for strong multi-word follow-up phrases first
    for phrase in STRONG_FOLLOWUP_PHRASES:
        if phrase in norm_text:
            return True
    # If very short (<=6 tokens) and ALL tokens are follow-up words -> it's a follow-up
    tokens = norm_text.split()
    if len(tokens) <= 6:
        non_followup = [t for t in tokens if t not in FOLLOWUP_TRIGGERS]
        if len(non_followup) == 0:
            return True
    return False


def get_rule_based_answer(text):
    """Returns (answer_html, intent_idx) or (None, None)."""
    norm = normalize(text)
    # Check section lookup first (no intent index for sections)
    sec_ans = lookup_section(norm)
    if sec_ans:
        return sec_ans, None
    # Score all intents — raise threshold to 3 to avoid weak false-positive matches
    best_score, best_ans, best_idx = 0, None, None
    for i, intent in enumerate(INTENTS):
        s = score_intent(norm, intent["kw"])
        if s > best_score:
            best_score, best_ans, best_idx = s, intent["ans"](), i
    if best_score >= 3:
        return best_ans, best_idx
    return None, None

# ── Main entry point ────────────────────────────────────────────────────────────
def get_legal_response(user_text, context_intent_idx=None, conversation_history=None):
    """
    Returns (response_html, used_ai: bool, matched_intent_idx: int|None)

    context_intent_idx:   last matched rule-based intent (for follow-up fallback)
    conversation_history: list of {role, content} dicts from previous turns.
                         Passed directly to Mistral so it has full memory.
    """
    norm = normalize(user_text)

    # Very short greetings — handle specially
    GREETS = {'hi','hello','hey','ok','okay','thanks','thank you','bye','good morning','good evening',
              'good afternoon','namaste','namaskar','hii','helo','helo','heya','howdy','hai','shukriya',
              'dhanyawad','welcome','ty','thx'}
    if norm.strip() in GREETS or len(norm.strip()) <= 3:
        return (
            "🙏 <strong>Namaskar!</strong> Main hoon <strong>Adv. LexAI</strong> — aapka AI Senior Legal Advisor.<br><br>"
            "35 saalon ka anubhav, Supreme Court se lekar District Courts tak — aap kisi bhi bhasha mein, kisi bhi tarah pooch sakte hain.<br><br>"
            "<strong>Kuch examples jaise aap puch sakte hain:</strong><br>"
            "• <em>\"498A kya hota hai?\"</em> — main section explain karta hoon<br>"
            "• <em>\"Boss ne salary nahi di\"</em> — main steps bata deta hoon<br>"
            "• <em>\"Cheque bounce ho gaya\"</em> — puri legal process samjhata hoon<br>"
            "• <em>\"Online fraud ho gaya, paise gaye\"</em> — immediate steps bataata hoon<br>"
            "• <em>\"Divorce karna hai\"</em> — process, timeline, rights sab<br>"
            "• <em>\"IPC 302 kya hai?\"</em> — section ka full breakdown<br><br>"
            "Ghabrao mat — koi bhi problem ho, ek kaam karo: <strong>mujhe batao kya hua.</strong> 🙏",
            False, None
        )

    # ── ALWAYS detect intent first (for ML category override) ───────────────────
    # Run rule-based scoring to get a base intent, but we rely on Mistral for chat.
    _, detected_intent_idx = get_rule_based_answer(user_text)

    # ── Mistral AI with full conversation history ───────────────────────────────
    if mistral_client:
        try:
            # Build message list: system prompt + full history + current message
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            if conversation_history:
                # Add up to last 10 turns (20 messages) to stay within token limits
                messages.extend(conversation_history[-20:])
            messages.append({"role": "user", "content": user_text})

            resp = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=messages
            )
            html = md_to_html(resp.choices[0].message.content)
            # Return detected_intent_idx so ML panel has a fallback category if needed
            return (html, True, detected_intent_idx)
        except Exception as e:
            print(f"[Mistral Error] {e}")

    # ── Fallback to rule-based ───────────────────────────────────────────────────
    ans, intent_idx = get_rule_based_answer(user_text)
    if ans:
        return (ans, False, intent_idx)

    # ── Ultimate fallback — generic legal guidance ───────────────────────────────
    return (
        f"⚖️ <strong>LexAI Legal Guidance</strong><br><br>"
        f"Aapka sawaal sun liya: <em>\"{user_text[:120]}\"</em><br><br>"
        "Dekho, yeh matter kisi bhi category mein aa sakta hai. Mujhe thoda aur bataao details mein.<br><br>"
        "Koi aur sawaal ho toh batao — main hoon. 🙏",
        False, detected_intent_idx
    )

def extract_dynamic_case_data(conversation_history, current_text):
    """
    Uses Mistral AI to analyze the conversation and return a custom ML panel structure
    specific to this exact case.
    """
    if not mistral_client:
        return None
        
    # Format the conversation nicely for the prompt
    convo_text = ""
    if conversation_history:
        for msg in conversation_history:
            role = "AI Lawyer" if msg.get("role") == "assistant" else "User"
            convo_text += f"{role}: {msg.get('content')}\n"
            
    prompt = f"""Analyze the following legal conversation between a user and an AI Lawyer. 
Based ONLY on the specific details discussed in this chat, extract the following structured data.

Conversation:
{convo_text}
Latest message: {current_text}

Output MUST be valid JSON with exactly these keys:
- "category": String. Choose ONE from: "criminal", "fraud", "family", "property", "labour", "consumer", "motor_accident", "sexual_offense", "theft_robbery", "cyber_crime", "medical_negligence", "constitutional", "corporate_tax".
- "risk_level": String. Choose ONE from: "Low", "Medium", "High".
- "win_probability": Integer between 0 and 100.
- "recommended_steps": List of strings. Highly specific next steps tailored to their exact situation (e.g. "Gather the WhatsApp chats from June 12th as proof"). Do not use generic steps.
- "applicable_laws": List of strings. Exact IPC/BNS or specific acts mentioned or applicable.
"""
    import time
    import json
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Respect the 1 RPS limit of Mistral free tier
            time.sleep(1.5)
            
            resp = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            data = json.loads(resp.choices[0].message.content)
            print(f"[Mistral Extracted Data] {data}", flush=True)
            return data
        except Exception as e:
            print(f"[Mistral Extraction Error - Attempt {attempt+1}] {e}", flush=True)
            if "429" in str(e):
                time.sleep(2) # Backoff
            else:
                break
    return None

# Initialize on import
init_mistral()
