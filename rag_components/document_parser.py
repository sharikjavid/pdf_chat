from base64 import b64decode
from langchain.schema.document import Document
from typing import List, Dict, Union, Any

class DocumentParser:
    @staticmethod
    def parse_docs(docs: List[Union[str, Document, Any]]) -> Dict[str, List[Union[str, Document]]]:
        b64_images = []
        text_documents = []
        for i, doc in enumerate(docs):
            if isinstance(doc, str):
                try:
                    b64decode(doc, validate=True)
                    b64_images.append(doc)
                except Exception:
                    text_documents.append(Document(page_content=doc))
            elif isinstance(doc, Document):
                text_documents.append(doc)
            else:
                try:
                    content = str(doc) 
                    text_documents.append(Document(page_content=content))
                except Exception as e:
                    print(f"[WARN] document_parser.py: Could not parse doc {i} of type {type(doc)}: {e}")
                    continue
        return {"images": b64_images, "texts": text_documents}