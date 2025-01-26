from typing import Dict, Any, Optional
from embeddings import DocumentEmbedder
from document_processor import DocumentProcessor
from database import VectorDatabase
import os
import argparse

class HierarchicalRAG:
    def __init__(self):
        self.embedder = DocumentEmbedder()
        self.processor = DocumentProcessor(self.embedder)
        self.db = VectorDatabase()
    
    def index_folder(self, folder_path: str):
        """Index all documents in a folder"""
        print(f"Starting to index folder: {folder_path}")
        if not os.path.exists(folder_path):
            print(f"Error: Path {folder_path} does not exist!")
            return
            
        file_count = 0
        for doc_info in self.processor.process_folder(folder_path):
            self.db.insert_document(doc_info)
            file_count += 1
            print(f"Processed file {file_count}: {doc_info.metadata['file_name']}")
            
        self.db.collection.flush()
        print(f"Finished indexing {file_count} files")
    
    def search(self, 
               query: str, 
               folder_filters: Optional[Dict[str, Any]] = None, 
               limit: int = 5):
        """Search documents with filters"""
        print(f"Searching for: {query}")
        query_embedding = self.embedder.create_embeddings(query)
        results = self.db.search(query_embedding, folder_filters, limit)
        return results

def setup_argparse():
    parser = argparse.ArgumentParser(
        description='Hierarchical Document Search System',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Index documents in a folder')
    index_parser.add_argument(
        'folder_path',
        type=str,
        help='Path to the folder containing documents'
    )
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search in indexed documents')
    search_parser.add_argument(
        'query',
        type=str,
        help='Search query'
    )
    search_parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help='Maximum folder depth to search in'
    )
    search_parser.add_argument(
        '--root-folder',
        type=str,
        default=None,
        help='Root folder to search in'
    )
    search_parser.add_argument(
        '--limit',
        type=int,
        default=5,
        help='Maximum number of results to return'
    )
    
    return parser

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    rag = HierarchicalRAG()
    
    if args.command == 'index':
        # Index documents
        print(f"\n=== Indexing Documents from {args.folder_path} ===")
        rag.index_folder(args.folder_path)
        
    elif args.command == 'search':
        # Prepare filters
        folder_filters = {}
        if args.max_depth is not None:
            folder_filters['folder_depth'] = {'$lte': args.max_depth}
        if args.root_folder:
            folder_filters['root_folder'] = args.root_folder
            
        # Perform search
        print(f"\n=== Searching for: {args.query} ===")
        if folder_filters:
            print("Filters:", folder_filters)
            
        results = rag.search(
            query=args.query,
            folder_filters=folder_filters if folder_filters else None,
            limit=args.limit
        )
        
        # Print results
        print("\n=== Search Results ===")
        if not results:
            print("No results found")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Path: {result['path']}")
            print(f"Similarity: {result['similarity']:.3f}")
            print(f"Preview: {result['content'][:200]}...")

if __name__ == "__main__":
    main()
