import os
from dotenv import load_dotenv
from models.document_loader import DocumentLoader
from models.embedding_model import EmbeddingModel
from repositories.qdrant_repository import QdrantRepository
from models.prompt_generator import PromptGenerator
from services.ollama_service import OllamaService
from services.application_service import Application

# Carregar variáveis do arquivo .env
load_dotenv()

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'documentos.json')
    
    # Recuperando configurações do .env
    QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "documentos")
    OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://host.docker.internal:11434/api/generate")
    
    loader = DocumentLoader(json_path)
    embedder = EmbeddingModel()
    repository = QdrantRepository(
        host=QDRANT_HOST, port=QDRANT_PORT, collection_name=COLLECTION_NAME)
    prompt_generator = PromptGenerator()
    ollama_service = OllamaService(api_url=OLLAMA_API_URL)
    
    app = Application(loader, embedder, repository, prompt_generator, ollama_service)
    app.run()