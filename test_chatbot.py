"""Fast test - tests rule-based scoring only (no Mistral API calls)."""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, '.')


# Patch out Mistral so tests run fast
import legal_brain
legal_brain.mistral_client = None  # disable Mistral for fast testing

from legal_brain import (
    get_legal_response, get_rule_based_answer,
    INTENT_TO_CATEGORY, normalize, is_followup_query, INTENTS
)

print("\n" + "=" * 95)
print("INTENT DETECTION + CATEGORY MAPPING TEST")
print("=" * 95)
print(f"{'QUERY':<46} {'IDX':<5} {'CATEGORY':<18} {'RESPONSE_PREVIEW'}")
print("-" * 95)

queries = [
    ("hit and run case i am the victim",   None),
    ("mere ghar pe chori hui hai",         None),
    ("mera cheque bounce ho gaya kya karu",None),
    ("pati mujhe maar raha hai 498a",      None),
    ("online fraud ho gaya paise gaye",    None),
    ("boss ne salary nahi di 3 months",    None),
    ("divorce karna hai mutual",           None),
    ("bail kaise milegi arrested",         None),
    ("consumer complaint amazon refund",   None),
    ("property dispute landlord eviction", None),
]

for query, ctx in queries:
    _, intent_idx = get_rule_based_answer(query)
    cat = INTENT_TO_CATEGORY.get(intent_idx, "NO MATCH") if intent_idx is not None else "NO MATCH"
    intent_name = INTENTS[intent_idx]["kw"][0] if intent_idx is not None else "none"
    ok = "OK" if cat != "NO MATCH" else "FAIL"
    print(f"[{ok}] {query[:44]:<44}  {str(intent_idx):<5} {cat:<18} ({intent_name})")

print("\n" + "=" * 95)
print("FOLLOW-UP DETECTION TEST")
print("=" * 95)
follow_up_cases = [
    ("guide me", True),
    ("detailed guidance chahiye", True),
    ("step by step kya karu", True),
    ("kya karu ab", True),
    ("aur batao", True),
    ("help me", True),
    ("hit and run case i am victim", False),     # real query, NOT follow-up
    ("mere ghar pe chori hui hai", False),        # real query, NOT follow-up
    ("pati maar raha hai 498a", False),           # real query, NOT follow-up
]

all_passed = True
for phrase, expected in follow_up_cases:
    n = normalize(phrase)
    result = is_followup_query(n)
    ok = "PASS" if result == expected else "FAIL"
    if ok == "FAIL": all_passed = False
    print(f"  [{ok}] '{phrase}' -> expected={expected}, got={result}")

print("\n" + "=" * 95)
print("CONTEXT FOLLOW-UP RESPONSE TEST")
print("=" * 95)
# Simulate: user said "chori hui hai" (theft=21), then says "guide me"
html, used_ai, idx = get_legal_response("guide me", context_intent_idx=21)
cat = INTENT_TO_CATEGORY.get(idx, "none")
print(f"  'guide me' with theft context -> intent_idx={idx}, category={cat}")
ok = "PASS" if idx == 21 and "Chori" in html else "CHECK"
print(f"  [{ok}] Response starts with: {html[:80]}")

# Simulate: user said "hit and run" (0), then says "step by step"
html, used_ai, idx = get_legal_response("step by step kya karu", context_intent_idx=0)
cat = INTENT_TO_CATEGORY.get(idx, "none")
print(f"\n  'step by step' with hit-run context -> intent_idx={idx}, category={cat}")
ok = "PASS" if idx == 0 else "CHECK"
print(f"  [{ok}] Response starts with: {html[:80]}")

print("\n" + "=" * 95)
print("ALL TESTS COMPLETE")
print("=" * 95)
