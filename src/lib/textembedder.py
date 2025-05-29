from typing import List

from langchain.docstore.document import Document as LangchainDocument
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient


def embed_documents(documents: List[LangchainDocument],
                    embedding_model: HuggingFaceEmbeddings,
                    qdrant_host: str,
                    qdrant_port: int,
                    collection_name: str,
                    distance: str = 'Cosine') -> QdrantVectorStore:

    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    if not client.collection_exists(collection_name):
        return QdrantVectorStore.from_documents(
            documents=documents,
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
            embedding=embedding_model,
            distance=distance,
        )
    else:
        return QdrantVectorStore.from_existing_collection(
            host=qdrant_host,
            port=qdrant_port,
            collection_name=collection_name,
            embedding=embedding_model,
            distance=distance,
        )
