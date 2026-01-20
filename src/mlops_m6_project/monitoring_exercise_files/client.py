import requests

response = requests.post('http://localhost:8000/predict', json={"review": "I hate this product!"})
print(response.json())