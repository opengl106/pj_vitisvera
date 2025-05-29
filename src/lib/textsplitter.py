from typing import List, Optional

from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from tqdm import tqdm

from src.utils.consts.models import THENPER_GTE_SMALL
from src.utils.consts.separators import PDF_EBOOK_ENGLISH_SEPARATORS


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = THENPER_GTE_SMALL,
) -> List[LangchainDocument]:
    """
    Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=PDF_EBOOK_ENGLISH_SEPARATORS,
        is_separator_regex=True,
    )

    docs_processed = []
    for doc in tqdm(knowledge_base, desc="Splitting documents"):
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique
