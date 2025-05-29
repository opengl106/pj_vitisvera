import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

from src.lib.docloader import load_documents
from src.lib.textsplitter import split_documents

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

RAW_KNOWLEDGE_BASE = load_documents("./texts")

docs_processed = split_documents(
    512,
    RAW_KNOWLEDGE_BASE,
    tokenizer_name=EMBEDDING_MODEL_NAME,
)

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
fig = pd.Series(lengths).hist()
plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
plt.show()
