import json
import urllib.request
import urllib.parse
import urllib.error

print("Testing GET /api/search...")
try:
    url = 'http://127.0.0.1:5000/api/search?q=' + urllib.parse.quote('Aug')
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        data = json.loads(response.read().decode())
        print("Response snippet:", str(data)[:100])
except urllib.error.URLError as e:
    print("GET Error:", e)

print("\nTesting POST /api/analyze...")
try:
    payload = {
        'brand_names': ['Augmentin 625 Duo Tablet', 'Ascoril LS Syrup'],
        'patient_data': {
            'age': 72,
            'gender': 'Female',
            'conditions': ['Hypertension'],
            'lab_values': {'eGFR': 42}
        }
    }
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request('http://127.0.0.1:5000/api/analyze', data=data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        print("Status:", response.status)
        data = json.loads(response.read().decode())
        print("Keys returned:", list(data.keys()))
        if 'error' in data:
            print("API Error:", data['error'])
except urllib.error.URLError as e:
    print("POST Error:", e)

