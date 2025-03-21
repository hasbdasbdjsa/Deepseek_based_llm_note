import requests

API_URL = "http://localhost:8000/query"
data = {"text": "请介绍一下你自己"}

response = requests.post(API_URL, json=data)
print(response.json())