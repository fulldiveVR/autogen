"""Chroma-based implementation of the Library interface."""

from typing import List, Optional, Dict, Any, Union
from .library_base import Library
from .retriever import Document, ChromaDocumentRetriever

class ChromaLibrary(Library):
    """Chroma-based implementation of the Library interface."""
    
    def __init__(
        self,
        collection_name: str = "autogen_docs",
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """Initialize the Chroma library.
        
        Args:
            collection_name: Name of the Chroma collection to use.
            embedding_model: Name of the sentence-transformers model to use for embeddings.
            persist_directory: Directory to persist the database.
            chunk_size: Size of text chunks for splitting documents.
            chunk_overlap: Number of characters to overlap between chunks.
        """
        self.retriever = ChromaDocumentRetriever(
            collection_name=collection_name,
            embedding_model=embedding_model,
            persist_directory=persist_directory
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _chunk_document(self, doc: Document) -> List[Document]:
        """Split a document into chunks.
        
        Args:
            doc: Document to split.
            
        Returns:
            List of Document chunks.
        """
        # Simple character-based chunking for now
        content = doc.content
        chunks = []
        start = 0
        
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            if end < len(content) and not content[end].isspace():
                # Try to end at a space to avoid cutting words
                last_space = content.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_content = content[start:end]
            chunk_metadata = doc.metadata.copy() if doc.metadata else {}
            chunk_metadata.update({
                'chunk_index': len(chunks),
                'original_document_id': doc.id,
            })
            
            chunks.append(Document(
                content=chunk_content,
                metadata=chunk_metadata
            ))
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
        
        return chunks
    
    def add_documents(self, documents: Union[Document, List[Document]]) -> None:
        if isinstance(documents, Document):
            documents = [documents]
        
        all_chunks = []
        for doc in documents:
            chunks = self._chunk_document(doc)
            all_chunks.extend(chunks)
        
        self.retriever.add_documents(all_chunks)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        return self.retriever.retrieve(query, top_k, where)
    
    def clear(self) -> None:
        self.retriever.delete_collection()
