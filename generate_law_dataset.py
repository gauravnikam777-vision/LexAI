import json
import os
import random

# Advanced Indian Legal Knowledge Base Template
# This curates perfect legal advice with accurate IPC, BNS, and Act mappings.

templates = [
    {
        "category": "fraud",
        "scenarios": [
            "My debit card was cloned and money was withdrawn.",
            "Someone used my credit card for unauthorized international transactions.",
            "I became a victim of an online bank fraud phishing attack.",
            "OTP scam resulted in money being deducted from my savings account."
        ],
        "risk_level": "High",
        "outcome": "win",
        "ipc_sections": ["Section 420 IPC (Cheating)", "Section 318 BNS (Cheating)", "Section 66C and 66D of IT Act 2000"],
        "what_to_do": "Block the card/account instantly and report the financial fraud to authorities.",
        "how_to_do": [
            "Call your bank's 24/7 helpline to block the card/account immediately.",
            "Call the National Cyber Crime Helpline at 1930.",
            "Register a formal complaint at cybercrime.gov.in with transaction screenshots and bank statements.",
            "Submit a written dispute form to your bank manager within 3 days."
        ],
        "when_to_do": "URGENT. Must be reported within 72 hours to ensure zero liability under RBI guidelines."
    },
    {
        "category": "fraud",
        "scenarios": [
            "A cheque issued by a customer bounced due to insufficient funds.",
            "Business partner gave a post-dated cheque which was dishonored.",
            "Cheque returned unpaid by bank marked as funds insufficient."
        ],
        "risk_level": "Medium",
        "outcome": "win",
        "ipc_sections": ["Section 138 of Negotiable Instruments Act 1881"],
        "what_to_do": "Send a legal demand notice to the issuer before filing a court case.",
        "how_to_do": [
            "Collect the 'Cheque Return Memo' from your bank.",
            "Draft a formal Legal Demand Notice demanding payment (preferably through a lawyer).",
            "Send the notice via Registered Post / Speed Post with acknowledgment due.",
            "If they fail to pay after receiving the notice, file a criminal complaint in the Magistrate court."
        ],
        "when_to_do": "Notice MUST be sent within 30 days of receiving the return memo from bank. The drawer gets 15 days to pay. Case must be filed within 30 days after that."
    },
    {
        "category": "property",
        "scenarios": [
            "My tenant is refusing to vacate my house after the lease agreement expired.",
            "Landlord eviction dispute where tenant has not paid rent for 6 months and won't leave.",
            "Tenant illegally subletting the property and refusing to move out."
        ],
        "risk_level": "High",
        "outcome": "win",
        "ipc_sections": ["Specific Relief Act 1963", "State Rent Control Act", "Section 441 IPC (Criminal Trespass) if applicable"],
        "what_to_do": "Initiate legal eviction proceedings. Do NOT take the law into your own hands by cutting electricity or water.",
        "how_to_do": [
            "Send a strict legal notice for eviction terminating the tenancy.",
            "Do not forcibly evict them or cut essential services, as that is illegal.",
            "If they do not vacate, file an eviction suit in the local Rent Controller Court or Civil Court.",
            "Present the registered rent agreement and bank statements showing unpaid rent."
        ],
        "when_to_do": "Send the legal notice immediately after the lease expires. Allow 1 month notice period."
    },
    {
        "category": "property",
        "scenarios": [
            "Landlord is refusing to return my security deposit after I vacated the flat.",
            "Withholding of security advance deposit without any valid proof of damage to the rented property.",
            "Owner not refunding deposit money despite leaving the house in perfect condition."
        ],
        "risk_level": "Medium",
        "outcome": "win",
        "ipc_sections": ["Indian Contract Act 1872", "Section 403 IPC (Dishonest misappropriation)"],
        "what_to_do": "Send a written demand, followed by legal action for recovery.",
        "how_to_do": [
            "Send formal emails/messages asking for the refund to create a paper trail.",
            "Send a legal notice via a lawyer threatening civil/criminal action for breach of contract.",
            "If unresolved, file a police complaint for criminal breach of trust, or approach a civil court for recovery."
        ],
        "when_to_do": "Wait for the agreed refund period (usually 15-30 days), then send notice immediately."
    },
    {
        "category": "property",
        "scenarios": [
            "Neighbor has constructed a wall that encroaches upon my land.",
            "Boundary dispute where adjacent plot owner is capturing my property legally.",
            "Illegal encroachment on private land without permission."
        ],
        "risk_level": "High",
        "outcome": "win",
        "ipc_sections": ["Section 441 IPC (Criminal Trespass)", "Section 329 BNS (Criminal Trespass)", "Specific Relief Act 1963"],
        "what_to_do": "File for an urgent injunction and officially measure the land.",
        "how_to_do": [
            "Apply to the local Tehsildar or Revenue Dept for official measurement and demarcation of land.",
            "File an FIR at the local police station for criminal trespass.",
            "Approach the Civil Court to get a 'Stay Order' (temporary injunction) to stop the construction immediately."
        ],
        "when_to_do": "Extremely urgent. Take legal action the moment encroachment begins to prevent permanent structures."
    },
    {
        "category": "labour",
        "scenarios": [
            "My employer terminated me suddenly without giving mandatory notice period or severance pay.",
            "Fired from company without any prior warning or compensation after years of service.",
            "Wrongful termination by HR without following proper retrenchment procedures."
        ],
        "risk_level": "High",
        "outcome": "win",
        "ipc_sections": ["Industrial Disputes Act 1947", "State Shops and Establishments Act"],
        "what_to_do": "Challenge the termination and claim your rightful severance and pending dues.",
        "how_to_do": [
            "Review your offer letter and company policy for notice period clauses.",
            "Write a formal grievance email to HR outlining illegal termination.",
            "Send a legal notice claiming outstanding salary, leave encashment, and severance.",
            "File a complaint with the District Labour Commissioner if the company ignores the notice."
        ],
        "when_to_do": "Send the grievance email immediately. Issue a legal notice within 15-30 days of termination."
    },
    {
        "category": "labour",
        "scenarios": [
            "Senior manager is sexually harassing me in the workplace.",
            "Sexual harassment at office by colleague, HR is not taking my complaint seriously.",
            "Hostile work environment created by inappropriate sexual advances from boss."
        ],
        "risk_level": "High",
        "outcome": "win",
        "ipc_sections": ["POSH Act 2013", "Section 354A IPC (Sexual Harassment)", "Section 74 BNS (Sexual Harassment)"],
        "what_to_do": "Report and escalate immediately strictly following the POSH guidelines.",
        "how_to_do": [
            "File a formal written complaint to the Internal Complaints Committee (ICC) of the company.",
            "Preserve all evidence (emails, WhatsApp chats, witness statements).",
            "If the company lacks an ICC or ignores it, file a complaint with the Local Complaints Committee (LCC) at the district level.",
            "File a zero FIR at the nearest police station under Section 354A."
        ],
        "when_to_do": "Must file the ICC complaint within 3 months of the last incident occurring."
    },
    {
        "category": "labour",
        "scenarios": [
            "Company is deducting Provident Fund (PF) from my salary but not depositing it into my EPFO account.",
            "Employer is not remitting my PF contributions causing financial loss.",
            "EPF fraud by employer over several years."
        ],
        "risk_level": "High",
        "outcome": "win",
        "ipc_sections": ["Employees Provident Funds and Miscellaneous Provisions Act 1952", "Section 405 IPC (Criminal Breach of Trust)"],
        "what_to_do": "Complain to the EPFO and initiate criminal proceedings against directors.",
        "how_to_do": [
            "Download your EPF passbook and gather all salary slips showing deductions.",
            "File an online grievance on the EPFiGMS portal (epfigms.gov.in).",
            "Submit a written complaint to the Regional PF Commissioner with evidence.",
            "Since PF default is a criminal offense, you can also file an FIR for criminal breach of trust."
        ],
        "when_to_do": "As soon as you notice the missing deposits on the portal."
    },
    {
        "category": "fraud",
        "scenarios": [
            "I paid the full advance amount to a real estate builder 3 years ago but the flat construction never started.",
            "Real estate developer disappeared after taking booking money for apartment.",
            "Builder fraud where possession of flat is delayed indefinitely without refund."
        ],
        "risk_level": "High",
        "outcome": "win",
        "ipc_sections": ["RERA Act 2016", "Consumer Protection Act 2019", "Section 420 IPC (Cheating)"],
        "what_to_do": "File a strong complaint with the real estate regulatory authority.",
        "how_to_do": [
            "Collect the builder-buyer agreement, payment receipts, and brochures.",
            "File an online complaint with your State's RERA (Real Estate Regulatory Authority) demanding possession with delay penalty, or a full refund with interest.",
            "Alternatively, approach the National Consumer Disputes Redressal Commission (NCDRC).",
            "File a parallel FIR against the builder for cheating and fraud."
        ],
        "when_to_do": "As soon as the builder breaches the promised handover date."
    },
    {
        "category": "family",  # New Category!
        "scenarios": [
            "My spouse wants a mutual consent divorce.",
            "We have decided to separate amicably and file for divorce.",
            "Filing for mutual divorce without contested litigation."
        ],
        "risk_level": "Low",
        "outcome": "win",
        "ipc_sections": ["Hindu Marriage Act 1955 (Sec 13B)", "Special Marriage Act 1954 (Sec 28)"],
        "what_to_do": "Draft a joint petition and terms of settlement.",
        "how_to_do": [
            "Hire a common lawyer to draft an MoU defining alimony, child custody, and property division.",
            "File a joint petition in the Family Court.",
            "Wait for the mandatory 6-month 'cooling off' period (can be waived by the Supreme Court).",
            "Appear for the second motion to get the final decree."
        ],
        "when_to_do": "You must be living separately for at least 1 year before filing."
    },
    {
        "category": "consumer",  # New Category!
        "scenarios": [
            "I bought a defective mobile phone and the company refuses to honor the warranty.",
            "E-commerce site delivered a fake product and rejected my return request.",
            "Car dealership sold me a faulty vehicle and service center is avoiding me."
        ],
        "risk_level": "Low",
        "outcome": "lose",
        "ipc_sections": ["Consumer Protection Act 2019"],
        "what_to_do": "File a formal complaint in a consumer court.",
        "how_to_do": [
            "Send a legal notice to the company giving them 15 days to refund/replace.",
            "Register a grievance on the National Consumer Helpline (NCH).",
            "If unresolved, file an e-Daakhil complaint online at the District Consumer Commission.",
            "Upload invoices, warranty cards, and communication proofs."
        ],
        "when_to_do": "Consumer complaint must be filed within 2 years from the date of the defect/issue."
    }
]

# Generate permutations to build a robust dataset
dataset = []
case_id = 1

for t in templates:
    for scenario in t["scenarios"]:
        row = {
            "case_id": case_id,
            "text": scenario,
            "category": t["category"],
            "risk_level": t["risk_level"],
            "outcome": t["outcome"],
            "ipc_sections": " | ".join(t["ipc_sections"]),
            "what_to_do": t["what_to_do"],
            "how_to_do": json.dumps(t["how_to_do"]),
            "when_to_do": t["when_to_do"]
        }
        dataset.append(row)
        case_id += 1

# Save as JSON and CSV
import csv
os.makedirs('data', exist_ok=True)

with open('data/advanced_cases.json', 'w') as f:
    json.dump(dataset, f, indent=4)

with open('data/advanced_cases.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=dataset[0].keys())
    writer.writeheader()
    writer.writerows(dataset)

print(f"✅ Generated Advanced Indian Legal Knowledge Base: {len(dataset)} prime scenarios.")
