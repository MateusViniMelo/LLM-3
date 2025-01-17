import requests
from typing import Optional

class OllamaService:
    def __init__(self, api_url: str):
        self.api_url = api_url

    def query(self, prompt: str) -> Optional[str]:
        headers = {"Content-Type": "application/json"}
        payload = {"model": "llama3", "prompt": prompt, "stream": False}

        response = requests.post(self.api_url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get("response")
        else:
            print(f"Erro ao consultar o Ollama: {response.text}")