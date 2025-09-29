import os, json
import httpx
from mistralai import Mistral

API_KEY = "b2k0eGGJMhnz363rUN2pL0eDRDSpkAqE"
AGENT_ID_1 = "ag:a30ad7d2:20250929:nl-convertor:b8093f9d" 
AGENT_ID_2 = "ag:a30ad7d2:20250929:agent-question:043afb30"

# Client HTTP INSECURE (bypass TLS verification)
http_client = httpx.Client(verify=False, timeout=30)

client = Mistral(api_key=API_KEY, client=http_client)

# JSON-only sur l’agent (facultatif)
client.beta.agents.update(
    agent_id=AGENT_ID_1,
    completion_args={"response_format": {"type": "json_object"}}
)

m = ("Nous sommes accompagné de deux enfants de 4 et 6 ans, "
     "nous souhaitons une visite culturelle a partir de 10h "
     "à un rythme tranquille avec une pause goûter à 16h")

resp = client.beta.conversations.start(
    agent_id=AGENT_ID_1,
    inputs=[
        {"role": "user", "content": m},
    ],
    store=False
)

text = resp.outputs[0].content if resp.outputs else ""
json_response = json.loads(text)
print("#############################AGENT 1 reponse :")
print(json_response)

missing_field = json_response['missing_fields']
interests =json_response['interests']
constraints =json_response['constraints']
meals_services =json_response['meals_services']
tickets_logistics =json_response['tickets_logistics']
language =json_response['language']
time_window = json_response['time_window']

input_ag_qu = {
    "language":language,
    "fields_to_ask":missing_field,
    "interests" : interests,
    "constraints":constraints,
    "meals_services":meals_services,
    "tickets_logistics":tickets_logistics,
    "time_window":time_window,
}


print("############################# INPUT AGENT 2")
m_str = str(input_ag_qu)
print(m_str)
# AGENT 2
client.beta.agents.update(
    agent_id=AGENT_ID_2,
    completion_args={"response_format": {"type": "text"}}
)



resp_2 = client.beta.conversations.start(
    agent_id=AGENT_ID_2,
    inputs=[
        {"role": "user", "content": m_str},
    ],
    store=False
)

text = resp_2.outputs[0].content if resp_2.outputs else ""
print("############################# agent 2 reponse")
print(text)
