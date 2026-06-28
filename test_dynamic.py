import requests
import time

base = 'http://127.0.0.1:5000'
s = requests.Session()

print("Registering/Logging in...")
s.post(base + '/register', data={'username': 'flow_test', 'password': 'password123', 'role': 'client'})
r_login = s.post(base + '/login', json={'username': 'flow_test', 'password': 'password123'})
print("Login status:", r_login.status_code)

print("\n--- User Message 1 (Vague) ---")
r1 = s.post(base + '/api/analyze', json={'text': 'mere bhai ko police ne pakda hai'})
print("Status:", r1.status_code)
d1 = r1.json()
print("Chatbot Reply:", d1.get('message', '').replace('<br>', '\n'))
print("ML Panel Data (should be somewhat generic/empty if no facts):")
if d1.get('has_ml'):
    print("Category:", d1.get('category'))
    print("Steps:", d1.get('strategy', {}).get('steps', []))

print("\n--- User Message 2 (Specifics) ---")
time.sleep(1) # simulate think
r2 = s.post(base + '/api/analyze', json={'text': 'wo kal raat se station mein hai, unhone reason bhi nahi bataya. hume use milne bhi nahi de rahe hain. uska phone bhi chheen liya.'})
print("Status:", r2.status_code)
d2 = r2.json()
print("Chatbot Reply:", d2.get('message', '').replace('<br>', '\n'))
print("ML Panel Data (should now be highly specific):")
if d2.get('has_ml'):
    print("Category:", d2.get('category'))
    print("Risk:", d2.get('risk_level'))
    print("Applicable Laws:", d2.get('strategy', {}).get('applicable_laws', []))
    print("Steps:", d2.get('strategy', {}).get('steps', []))
