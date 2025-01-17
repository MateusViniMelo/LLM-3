import os
import requests
import json
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# Carregar o modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Configuração do cliente Qdrant
qdrant_client = QdrantClient("http://localhost:6333")
collection_name = "documents_2"

# Carregar documentos do arquivo JSON
def carregar_documentos():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(base_dir, 'documentos.json')

        with open(json_path, 'r', encoding='utf-8') as file:
            documentos = json.load(file)

            # Validar estrutura dos documentos
            if not isinstance(documentos, list) or not all('pergunta' in doc and 'resposta' in doc for doc in documentos):
                raise ValueError("O arquivo JSON deve ser uma lista de objetos com as chaves 'pergunta' e 'resposta'.")

            # Transformar em um array de perguntas e respostas
            return [{"pergunta": doc["pergunta"], "resposta": doc["resposta"]} for doc in documentos]
    except FileNotFoundError:
        print("Erro: O arquivo 'documentos.json' não foi encontrado.")
        return []
    except json.JSONDecodeError:
        print("Erro: O arquivo 'documentos.json' não contém um JSON válido.")
        return []
    except ValueError as e:
        print(f"Erro de validação: {e}")
        return []

documentos = carregar_documentos()

# Criar coleção no Qdrant caso não exista
if not qdrant_client.collection_exists(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "size": 384,
            "distance": "Cosine"
        }
    )

# Inserir documentos no Qdrant
for i, doc in enumerate(documentos):
    embedding = model.encode(doc["resposta"]).tolist()
    point = PointStruct(id=i, vector=embedding, payload={
        "text": doc["resposta"]})
    qdrant_client.upsert(collection_name=collection_name, points=[point])

print(f"Documentos armazenados na coleção '{collection_name}'.")

# Função para buscar documentos no Qdrant
def search_documents(query, model, client, collection_name, k=3):
    query_embedding = model.encode(query).tolist()
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k
    )
    return [{"pergunta": doc.payload.get("pergunta", "Pergunta não encontrada"), 
             "resposta": doc.payload["text"]} for doc in search_result]


# Função para criar o prompt
def create_prompt(documents, query):
    template = """
    Você é um assistente que responde com base nos seguintes documentos:
    {documents}

    Pergunta do usuário: {query}

    Resposta:
    """
    return template.format(documents="\n".join([f"P: {doc['pergunta']}\nR: {doc['resposta']}" for doc in documents]), query=query)

# Função para consultar o Ollama
def query_ollama(prompt):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"model": "llama3", "prompt": prompt, "stream": False}

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        print(f"Erro ao consultar o Ollama: {response.text}")
        return None

# Interação com o usuário no terminal
def main():
    while True:
        # Receber a pergunta do usuário
        query = input("Digite sua pergunta (ou 'sair' para finalizar): ")
        
        if query.lower() == 'sair':
            print("Saindo...")
            break
        
        # Buscar documentos relevantes
        retrieved_docs = search_documents(query, model, qdrant_client, collection_name)
        prompt = create_prompt(retrieved_docs, query)
        
        # Consultar o Ollama
        response = query_ollama(prompt)
        
        if response:
            print("Resposta final:", response)
        else:
            print("Não foi possível gerar uma resposta.")

if __name__ == "__main__":
    main()
