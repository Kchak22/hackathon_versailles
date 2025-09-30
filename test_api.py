import requests

url = "https://hackversailles-2-swds.ngrok.app/chat"
payload = {"question": "Bonjour, que me conseilles-tu à Versailles demain ?"}

try:
    r = requests.post(url, json=payload, timeout=60)
    print("Status code:", r.status_code)
    print("Response JSON:", r.json())
except Exception as e:
    print("Erreur lors de l'appel API:", e)
