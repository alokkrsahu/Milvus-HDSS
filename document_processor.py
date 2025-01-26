## document_processor.py
from pathlib import Path
import os
import numpy as np
from typing import Dict, Any, Generator
from dataclasses import dataclass
import magic  # for file type detection
from embeddings import DocumentEmbedder
import textract  # for text extraction from various formats

@dataclass
class DocumentInfo:
    content: str
    metadata: Dict[str, Any]
    embedding: np.ndarray = None

class DocumentProcessor:
    def __init__(self, embedder: DocumentEmbedder):
        self.embedder = embedder
        self.supported_extensions = {'.txt', '.pdf', '.docx', '.doc', '.md'}
    
    def process_folder(self, root_path: str) -> Generator[DocumentInfo, None, None]:
        """Process all documents in folder hierarchy"""
        root = Path(root_path)
        
        for file_path in root.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc_info = self._process_file(file_path, root)
                    if doc_info:
                        yield doc_info
                except Exception as e:
                    print(f"Error processing {file_path}: {str(e)}")
    
    def _process_file(self, file_path: Path, root_path: Path) -> DocumentInfo:
        """Process individual file"""
        # Extract text
        content = textract.process(str(file_path)).decode('utf-8')
        
        # Create relative path from root
        rel_path = file_path.relative_to(root_path)
        
        # Build metadata
        metadata = {
            'file_name': file_path.name,
            'file_type': magic.from_file(str(file_path), mime=True),
            'file_size': file_path.stat().st_size,
            'folder_hierarchy': list(rel_path.parent.parts),
            'folder_depth': len(rel_path.parent.parts),
            'root_folder': rel_path.parts[0] if rel_path.parts else '',
            'parent_folder': rel_path.parent.name,
            'full_path': str(rel_path)
        }
        
        # Create embedding
        embedding = self.embedder.create_embeddings(content)
        
        return DocumentInfo(content=content, metadata=metadata, embedding=embedding)
