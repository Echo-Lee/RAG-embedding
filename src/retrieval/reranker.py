"""Reranking module for improving retrieval results"""
from typing import List, Dict
import numpy as np
from sentence_transformers import CrossEncoder


class BaseReranker:
    """Base class for rerankers"""

    def rerank(self, query: str, documents: List[Dict], top_k: int) -> List[Dict]:
        """
        Rerank documents based on query

        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top documents to return

        Returns:
            Reranked list of documents
        """
        raise NotImplementedError


class CrossEncoderReranker(BaseReranker):
    """Cross-Encoder based reranker using sentence-transformers"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize Cross-Encoder reranker

        Args:
            model_name: Name or path of cross-encoder model
        """
        print(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        print("Reranker loaded successfully")

    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        Rerank documents using Cross-Encoder

        Args:
            query: Search query
            documents: List of retrieved documents with 'content' key
            top_k: Number of top documents to return

        Returns:
            Reranked documents with added 'rerank_score' field
        """
        if not documents:
            return []

        # Prepare query-document pairs
        pairs = [[query, doc['content']] for doc in documents]

        # Compute relevance scores
        scores = self.model.predict(pairs)

        # Sort by scores and get top_k
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        # Build reranked results
        reranked_docs = []
        for idx in ranked_indices:
            doc = documents[idx].copy()
            doc['rerank_score'] = float(scores[idx])
            reranked_docs.append(doc)

        return reranked_docs


class NoReranker(BaseReranker):
    """Dummy reranker that returns documents as-is"""

    def rerank(self, query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
        """Return top_k documents without reranking"""
        return documents[:top_k]
