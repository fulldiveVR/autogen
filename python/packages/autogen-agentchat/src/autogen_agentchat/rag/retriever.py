"""Document retriever for RAG capabilities using Chroma as vector store."""

from typing import List, Optional, Dict, Any, Union
import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dataclasses import dataclass
import uuid

@dataclass
class Document:
    """A document with its content and metadata."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.metadata is None:
            self.metadata = {}

class ChromaDocumentRetriever:
    """Document retriever using Chroma as the vector store backend."""
    
    def __init__(
        self,
        collection_name: str = "autogen_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
    ):
        """Initialize the Chroma document retriever.
        
        Args:
            collection_name: Name of the Chroma collection to use.
            embedding_model: Name of the sentence-transformers model to use for embeddings.
            persist_directory: Directory to persist the Chroma database. If None, uses in-memory database.
        """
        # Initialize Chroma client
        settings = Settings()
        if persist_directory:
            settings = Settings(persist_directory=persist_directory, anonymized_telemetry=False)
        
        self.client = chromadb.Client(settings)
        
        # Initialize embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_documents(self, documents: Union[Document, List[Document]]) -> None:
        """Add documents to the retriever.
        
        Args:
            documents: Document or list of documents to add.
        """
        if isinstance(documents, Document):
            documents = [documents]
            
        # Prepare documents for Chroma
        ids = [doc.id for doc in documents]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Add to collection
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve the most relevant documents for a query.
        
        Args:
            query: The query text.
            top_k: Number of documents to retrieve.
            where: Optional filter criteria for metadata.
            
        Returns:
            List of most relevant documents.
        """
        # Query the collection
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where
        )
        
        # Convert results to Documents
        documents = []
        for i in range(len(results['ids'][0])):
            doc = Document(
                content=results['documents'][0][i],
                metadata=results['metadatas'][0][i],
                id=results['ids'][0][i]
            )
            documents.append(doc)
            
        return documents
    
    def delete_collection(self) -> None:
        """Delete the current collection."""
        self.client.delete_collection(self.collection.name)
