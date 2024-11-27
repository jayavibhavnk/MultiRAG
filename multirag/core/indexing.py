import faiss
import numpy as np
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    text: Optional[str] = None
    image_path: Optional[str] = None
    text_embedding: Optional[np.ndarray] = None
    image_embedding: Optional[np.ndarray] = None

class MultiModalIndex:
    def __init__(self, text_dimension: int, image_dimension: int):
        self.text_index = faiss.IndexFlatL2(text_dimension)
        self.image_index = faiss.IndexFlatL2(image_dimension)
        self.documents: Dict[str, Document] = {}
        
    def add_document(self, doc: Document):
        if doc.text_embedding is not None:
            self.text_index.add(np.array([doc.text_embedding]))
        if doc.image_embedding is not None:
            self.image_index.add(np.array([doc.image_embedding]))
        self.documents[doc.id] = doc
