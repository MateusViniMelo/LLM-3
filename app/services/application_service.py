from models.document_loader import DocumentLoader
from models.embedding_model import EmbeddingModel
from repositories.qdrant_repository import QdrantRepository
from models.prompt_generator import PromptGenerator
from services.ollama_service import OllamaService


class Application:
    def __init__(
        self,
        loader: DocumentLoader,
        embedder: EmbeddingModel,
        repository: QdrantRepository,
        prompt_generator: PromptGenerator,
        ollama_service: OllamaService
    ):
        self.loader = loader
        self.embedder = embedder
        self.repository = repository
        self.prompt_generator = prompt_generator
        self.ollama_service = ollama_service

    def run(self) -> None:
        documents = self.loader.load_documents()
        if not documents:
            return

        self.repository.create_collection_if_not_exists(vector_size=384)
        self.repository.upsert_documents(documents, self.embedder)

        while True:
            query = input("Digite sua pergunta (ou 'sair' para finalizar): ")

            if query.lower() == 'sair':
                print("Saindo...")
                break

            query_vector = self.embedder.encode(query)
            retrieved_docs = self.repository.search(query_vector)
            prompt = self.prompt_generator.create_prompt(retrieved_docs, query)
            response = self.ollama_service.query(prompt)

            if response:
                print("\nResposta gerada:", response)
            else:
                print("\nNão foi possível gerar uma resposta.")
