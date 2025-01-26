from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from typing import List, Dict, Optional, Any
import numpy as np
from document_processor import DocumentInfo  

class VectorDatabase:
    def __init__(self, collection_name: str = "hierarchical_docs"):
        self.collection_name = collection_name
        connections.connect(host='localhost', port='19530')
        self._setup_collection()
    
    def _setup_collection(self):
        """Setup Milvus collection with necessary fields"""
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="folder_depth", dtype=DataType.INT64),
            FieldSchema(name="root_folder", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="parent_folder", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="full_path", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535)
        ]
        
        schema = CollectionSchema(fields=fields)
        
        if not utility.has_collection(self.collection_name):
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create indices
            self.collection.create_index(
                field_name="embedding",
                index_params={
                    "metric_type": "IP",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 1024}
                }
            )
            
            # Create scalar field indices
            self.collection.create_index(field_name="folder_depth")
            self.collection.create_index(field_name="root_folder")
            self.collection.create_index(field_name="parent_folder")
        else:
            self.collection = Collection(self.collection_name)
        
        self.collection.load()
    
    def insert_document(self, doc_info: DocumentInfo):
        """Insert document into database"""
        entity = [
            [doc_info.embedding.tolist()],
            [doc_info.metadata['folder_depth']],
            [doc_info.metadata['root_folder']],
            [doc_info.metadata['parent_folder']],
            [doc_info.metadata['full_path']],
            [doc_info.content]
        ]
        
        self.collection.insert(entity)
    
    def _build_filter_expression(self, folder_filters: Dict[str, Any]) -> str:
        """Build Milvus filter expression from filter dict"""
        if not folder_filters:
            return None
            
        expressions = []
        for field, condition in folder_filters.items():
            if isinstance(condition, dict):
                # Handle complex conditions (e.g., {'$lte': 3})
                for op, value in condition.items():
                    if op == '$lte':
                        expressions.append(f"{field} <= {value}")
                    elif op == '$gte':
                        expressions.append(f"{field} >= {value}")
                    elif op == '$lt':
                        expressions.append(f"{field} < {value}")
                    elif op == '$gt':
                        expressions.append(f"{field} > {value}")
            else:
                # Handle exact match
                if isinstance(condition, str):
                    expressions.append(f"{field} == '{condition}'")
                else:
                    expressions.append(f"{field} == {condition}")
                    
        return " && ".join(expressions) if expressions else None
    
    def search(self, 
               query_vector: np.ndarray,
               folder_filters: Optional[Dict[str, Any]] = None,
               limit: int = 5) -> List[Dict[str, Any]]:
        """Search with combined filtering and vector similarity"""
        expr = self._build_filter_expression(folder_filters)
        
        results = self.collection.search(
            data=[query_vector.tolist()],
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 10}},
            expr=expr,
            limit=limit,
            output_fields=["full_path", "content", "folder_depth", "root_folder"]
        )
        
        return [
            {
                "content": hit.entity.get('content'),
                "path": hit.entity.get('full_path'),
                "similarity": hit.score,
                "metadata": {
                    "folder_depth": hit.entity.get('folder_depth'),
                    "root_folder": hit.entity.get('root_folder')
                }
            }
            for hit in results[0]
        ]
