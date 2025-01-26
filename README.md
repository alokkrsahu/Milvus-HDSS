# Hierarchical Document Search System with Milvus Vector Database

## Project Overview
This project implements a hierarchical document search system using vector embeddings and the Milvus vector database. The system can process documents from nested folder structures while preserving their hierarchical relationships, create semantic embeddings, and enable efficient similarity search with folder-based filtering.

## Core Components and Functionality

### Document Embedding (embeddings.py)
The DocumentEmbedder class serves as the semantic understanding component of the system. It utilizes the Sentence Transformers library, specifically the 'all-MiniLM-L6-v2' model, to convert text into high-dimensional vector representations. This model is particularly effective for semantic search applications as it captures the meaning and context of text rather than just keywords.

### Document Processing (document_processor.py)
The DocumentProcessor handles the extraction and processing of documents from the file system. It supports multiple document formats (.txt, .pdf, .docx, .doc, .md) using the textract library for text extraction and python-magic for file type detection. The processor maintains the hierarchical structure of documents by preserving folder relationships and metadata such as folder depth, path information, and file attributes.

### Vector Database Management (database.py)
The VectorDatabase class manages all interactions with the Milvus vector database. It creates and maintains a collection with necessary fields for both vector similarity search and metadata filtering. The database schema includes:
- Document embeddings (384-dimensional vectors)
- Folder structure information (depth, root folder, parent folder)
- Content and path information
- Scalar indices for efficient filtering

### Main Application Interface (main.py)
The main application provides a command-line interface with two primary operations:
1. Document Indexing: Process and store documents from a specified folder hierarchy
2. Semantic Search: Query the database with natural language, applying optional folder-based filters

## Key Libraries and Technologies

### Milvus Vector Database
Milvus is an open-source vector database built specifically for handling large-scale vector similarity search. Key features:
- High-performance similarity search
- Hybrid search capabilities (vector similarity + scalar filtering)
- Scalable architecture
- Support for multiple index types
- Real-time data insertion and search

### Sentence Transformers
The project uses the Sentence Transformers library for creating semantic embeddings. The 'all-MiniLM-L6-v2' model provides a good balance between performance and accuracy for general-purpose text embeddings.

## Installation and Setup

### Prerequisites
1. Python 3.8 or higher
2. Docker and Docker Compose

### Installation Steps

1. Clone the repository and create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install required Python packages:
```bash
pip install pymilvus sentence-transformers textract python-magic numpy
```

3. Install system dependencies (Ubuntu/Debian):
```bash
sudo apt-get install python3-dev libxml2-dev libxslt1-dev antiword unrtf poppler-utils tesseract-ocr flac ffmpeg lame libmad0 libsox-fmt-mp3 sox libjpeg-dev swig libpulse-dev
```

4. Start Milvus using Docker Compose:
```bash
mkdir -p volumes/etcd volumes/minio volumes/milvus
docker-compose up -d
```

## Usage Guide

### Indexing Documents
To index documents from a folder:
```bash
python main.py index /path/to/your/documents
```

### Searching Documents
Basic search:
```bash
python main.py search "your search query"
```

Search with filters:
```bash
python main.py search "your query" --max-depth 3 --root-folder "technical" --limit 5
```

## System Architecture
The system follows a modular architecture where each component handles a specific responsibility:
1. The DocumentEmbedder creates semantic representations of text
2. The DocumentProcessor handles file system operations and metadata extraction
3. The VectorDatabase manages data storage and retrieval
4. The main application orchestrates these components and provides a user interface

The use of Milvus as the vector database enables efficient similarity search combined with metadata filtering, making it possible to search through large document collections while respecting folder hierarchies and organizational structures.

This architecture provides several benefits:
- Scalability: Can handle large document collections
- Flexibility: Supports various document formats and folder structures
- Performance: Efficient vector similarity search with metadata filtering
- Maintainability: Clear separation of concerns between components

