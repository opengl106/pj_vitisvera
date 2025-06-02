from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model(embedding_model_name: str,
                           normalize_embeddings: bool = True) -> HuggingFaceEmbeddings:

    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        multi_process=True,
        model_kwargs={"device": "cuda"},
        encode_kwargs={"normalize_embeddings": normalize_embeddings},  # Set `True` for cosine similarity
    )