from typing import List

from langchain_core.documents import Document as LangchainDocument
from transformers import AutoTokenizer


prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context,
give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
    },
]

def chat_template_generator(tokenizer: AutoTokenizer) -> str:
    return tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )


def context_generator(retrieved_docs: List[LangchainDocument]) -> str:
    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])
    return context
