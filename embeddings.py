# embeddings.py
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import numpy as np

class DocumentEmbedder:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.vector_dim = self.model.get_sentence_embedding_dimension()
    
    def create_embeddings(self, text: str) -> np.ndarray:
        """Create embeddings for document text"""
        return self.model.encode(text)
    
    def batch_create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for multiple texts in batch"""
        return self.model.encode(texts, batch_size=32, show_progress_bar=True)
