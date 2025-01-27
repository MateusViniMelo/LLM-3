import os
from models.document_loader import DocumentLoader
from models.embedding_model import EmbeddingModel
from repositories.qdrant_repository import QdrantRepository
from models.prompt_generator import PromptGenerator
from services.ollama_service import OllamaService
from services.application_service import Application


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, 'documentos.json')

    loader = DocumentLoader(json_path)
    embedder = EmbeddingModel()
    repository = QdrantRepository(
        host="localhost", port=6333, collection_name="documents_switch")
    prompt_generator = PromptGenerator()
    ollama_service = OllamaService(
        api_url="http://localhost:11434/api/generate")

    app = Application(loader, embedder, repository,
                      prompt_generator, ollama_service)
    app.run()
