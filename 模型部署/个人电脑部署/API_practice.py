import requests

API_URL = "http://localhost:8000/query"  #query即调用API，其中包含大模型的输出配置，按照该配置逐项生成
data = {"text": "请介绍一下你自己"}

response = requests.post(API_URL, json=data)  #此处可以与API.py文件中的response格式不同
print(response.json())