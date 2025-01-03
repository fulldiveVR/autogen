"""Base interface for RAG libraries."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from .retriever import Document

class Library(ABC):
    """Abstract base class for RAG libraries.
    
    This class defines the interface that all library implementations must follow.
    Different implementations can handle document processing, storage, and retrieval differently.
    """
    
    @abstractmethod
    def add_documents(self, documents: Union[Document, List[Document]]) -> None:
        """Add documents to the library.
        
        Args:
            documents: Document or list of documents to add.
        """
        pass
    
    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Retrieve relevant documents for a query.
        
        Args:
            query: The query text.
            top_k: Number of documents to retrieve.
            where: Optional filter criteria for metadata.
            
        Returns:
            List of most relevant documents.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all documents from the library."""
        pass
