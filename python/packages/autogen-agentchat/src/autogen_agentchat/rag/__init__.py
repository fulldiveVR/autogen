"""RAG (Retrieval Augmented Generation) capabilities for AutoGen agents.

Note: The RAGAgent class has been moved to autogen_agentchat.agents.
Import it using: from autogen_agentchat.agents import RAGAgent
"""

from .retriever import ChromaDocumentRetriever, Document
from .library import Library, ChromaLibrary

__all__ = ["ChromaDocumentRetriever", "Document", "Library", "ChromaLibrary"]
