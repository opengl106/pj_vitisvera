from src.lib.docloader import get_documents_paths, load_documents
from src.lib.textembedder import retrieve_embeddings
from src.lib.textgenerater import create_generater_pipeline
from src.lib.textsplitter import split_documents
from src.utils.consts.services import QDRANT_HOST, QDRANT_PORT
from src.utils.consts.models import THENLPER_GTE_SMALL, ZEPHYR_7B_BETA
from src.utils.templates.chat_template import chat_template_generator, context_generator
from src.utils.models.embedding import create_embedding_model
from src.utils.models.tokenizer import create_tokenizer


documents_paths = get_documents_paths("./texts")
RAW_KNOWLEDGE_BASE = load_documents(documents_paths)

docs_processed = split_documents(
    512,
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=THENLPER_GTE_SMALL,
)

"""
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

tokenizer = create_tokenizer(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
fig = pd.Series(lengths).hist()
plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
plt.show()
"""

embedding_model = create_embedding_model(THENLPER_GTE_SMALL)

"""
from src.lib.textembedder import embed_documents

KNOWLEDGE_VECTOR_DATABASE = embed_documents(
    documents=docs_processed,
    embedding_model=embedding_model,
    qdrant_host=QDRANT_HOST,
    qdrant_port=QDRANT_PORT,
    collection_name="demo_knowledge_base",
)
"""

KNOWLEDGE_VECTOR_DATABASE = retrieve_embeddings(
    embedding_model=embedding_model,
    qdrant_host=QDRANT_HOST,
    qdrant_port=QDRANT_PORT,
    collection_name="demo_knowledge_base",
)

user_query = "Ressentiment and Christianity"
query_vector = embedding_model.embed_query(user_query)
retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)

READER_LLM = create_generater_pipeline(ZEPHYR_7B_BETA)


tokenizer = create_tokenizer(ZEPHYR_7B_BETA)
RAG_PROMPT_TEMPLATE = chat_template_generator(tokenizer)
print(RAG_PROMPT_TEMPLATE)
context = context_generator(retrieved_docs)

final_prompt = RAG_PROMPT_TEMPLATE.format(question=user_query, context=context)

# Redact an answer
answer = READER_LLM(final_prompt)[0]["generated_text"]
print(answer)
