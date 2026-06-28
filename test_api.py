import requests
import sys

base = 'http://127.0.0.1:5000'
s = requests.Session()

print("Registering test user...")
r_reg = s.post(base + '/register', data={'username': 'api_tester', 'password': 'password123', 'role': 'client'})
print(f"Register status: {r_reg.status_code}")

print("Logging in...")
r_login = s.post(base + '/api/login', json={'username': 'api_tester', 'password': 'password123'})
print(f"Login status: {r_login.status_code}")
try:
    print(f"Login response: {r_login.json()}")
except:
    print(f"Login response text: {r_login.text[:100]}")

print("\nTesting case submission...")
r_cases = s.post(base + '/api/cases', json={
    'title': 'ghar par chori hui hai',
    'description': '3 chor aaye aur sab paise le gaye CCTV record hai'
})
print(f"Cases status: {r_cases.status_code}")
try:
    d = r_cases.json()
    print('Category:   ' + str(d.get('category')))
    print('Confidence: ' + str(d.get('cat_confidence')) + '%')
    print('Risk Level: ' + str(d.get('risk_level')))
    print('Case ID:    #' + str(d.get('case_id')))
except:
    print(f"Cases response text: {r_cases.text[:100]}")

print("\nTesting chatbot analyze...")
r_analyze = s.post(base + '/api/analyze', json={'text': 'ghar par chori hui hai kya karna chahiye'})
print(f"Analyze status: {r_analyze.status_code}")
try:
    d2 = r_analyze.json()
    print('Category:   ' + str(d2.get('category')))
    print('Confidence: ' + str(d2.get('cat_confidence')) + '%')
    msg = d2.get('message', '')
    print('Response preview: ' + msg[:100].replace('\n', ' ') + '...')
except:
    print(f"Analyze response text: {r_analyze.text[:100]}")
