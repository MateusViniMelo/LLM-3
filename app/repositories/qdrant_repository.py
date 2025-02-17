from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from typing import List, Dict
from models.embedding_model import EmbeddingModel
import uuid


class QdrantRepository:
    def __init__(self, host: str, port: int, collection_name: str):
        self.client = QdrantClient(f"http://{host}:{port}")
        self.collection_name = collection_name

    def create_collection_if_not_exists(self, vector_size: int, distance: str = "Cosine") -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={"size": vector_size, "distance": distance}
            )

    def upsert_documents(self, documents: List[Dict[str, str]], embedder: EmbeddingModel) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),  # Usando UUID para evitar conflitos de ID
                vector=embedder.encode(doc["resposta"]),
                payload={"pergunta": doc["pergunta"], "resposta": doc["resposta"]}
            )
            for doc in documents
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(self, query_vector: List[float], k: int = 3) -> List[Dict[str, str]]:
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k
        )
        return [
            {
                "pergunta": doc.payload.get("pergunta", "Pergunta não encontrada"),
                "resposta": doc.payload.get("resposta", "Resposta não encontrada"),
                "score": doc.score  # Retornando o score da similaridade
            }
            for doc in search_result
        ]
