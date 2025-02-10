import json
from typing import List, Dict


class DocumentLoader:
    def __init__(self, json_path: str):
        self.json_path = json_path

    def load_documents(self) -> List[Dict[str, str]]:
        try:
            with open(self.json_path, 'r', encoding='utf-8') as file:
                documents = json.load(file)
                if not isinstance(documents, list) or not all('pergunta' in doc and 'resposta' in doc for doc in documents):
                    raise ValueError(
                        "O arquivo JSON deve ser uma lista de objetos com as chaves 'pergunta' e 'resposta'.")
                return documents
        except FileNotFoundError:
            print("Erro: O arquivo JSON não foi encontrado.")
            return []
        except json.JSONDecodeError:
            print("Erro: O arquivo JSON não contém um JSON válido.")
            return []
        except ValueError as e:
            print(f"Erro de validação: {e}")
            return []
