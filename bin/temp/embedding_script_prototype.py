from src.lib.docloader import load_documents, get_documents_paths
from src.lib.textsplitter import split_documents

from src.utils.consts.models import THENPER_GTE_SMALL

documents_paths = get_documents_paths("./texts")
RAW_KNOWLEDGE_BASE = load_documents(documents_paths)

docs_processed = split_documents(
    512,
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=THENPER_GTE_SMALL,
)

"""
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
fig = pd.Series(lengths).hist()
plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
plt.show()
"""

from langchain_huggingface import HuggingFaceEmbeddings

from src.utils.consts.services import QDRANT_HOST, QDRANT_PORT
from src.lib.textembedder import embed_documents

embedding_model = HuggingFaceEmbeddings(
    model_name=THENPER_GTE_SMALL,
    multi_process=True,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
)

KNOWLEDGE_VECTOR_DATABASE = embed_documents(
    documents=docs_processed,
    embedding_model=embedding_model,
    qdrant_host=QDRANT_HOST,
    qdrant_port=QDRANT_PORT,
    collection_name="demo_knowledge_base",
    distance='Cosine',
)

user_query = "Ressentiment and Christianity"
query_vector = embedding_model.embed_query(user_query)
retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
