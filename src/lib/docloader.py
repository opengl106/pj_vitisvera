import os
from pathlib import PosixPath
from typing import List, Union

from langchain.docstore.document import Document as LangchainDocument

def load_documents(file_paths: List[PosixPath]) -> List[LangchainDocument]:
    documents = []
    for file in file_paths:
        with open(file, 'r') as f:
            text = f.read()
            documents.append(LangchainDocument(page_content=text, metadata={"title": file.name}))
    return documents

def get_documents_paths(files: Union[List[str], str]) -> List[PosixPath]:
    if isinstance(files, list):
        file_paths = [PosixPath(file) for file in files]
        return file_paths
    if os.path.isfile(files):
        return [PosixPath(files)]
    if os.path.isdir(files):
        file_paths = [PosixPath(files) / PosixPath(file) for file in os.listdir(files)]
        return file_paths
    raise ValueError(f"Invalid type for input files: {type(files)}")
