"""Retrieval module for document search"""
from typing import List, Dict, Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from .indexer import FAISSIndexBuilder
from .reranker import BaseReranker, NoReranker


class DenseRetriever:
    """Dense retrieval using FAISS"""

    def __init__(self, config):
        """
        Initialize dense retriever

        Args:
            config: RAGConfig instance
        """
        self.config = config
        self.model = None
        self.index = None
        self.doc_metadata = None

        # Load index and model
        self._load()

    def _load(self):
        """Load embedding model and FAISS index"""
        # Load embedding model
        print(f"Loading embedding model: {self.config.embedding_model}")

        if self.config.use_finetuned and self.config.finetuned_model_path:
            model_path = str(self.config.finetuned_model_path)
            print(f"Using fine-tuned model: {model_path}")
        else:
            model_path = self.config.embedding_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = self.config.max_seq_length

        print(f"Model loaded on device: {device}")

        # Load FAISS index
        indexer = FAISSIndexBuilder(self.config)
        self.index, self.doc_metadata = indexer.load_index()

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict]:
        """
        Retrieve top-k documents for a query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents with scores
        """
        # Encode query
        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        query_emb = np.asarray(query_emb, dtype=np.float32)

        # Search in FAISS
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_emb, k)

        # Build results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0:  # Invalid index
                continue

            doc = self.doc_metadata[idx]
            results.append({
                "content": doc["content"],
                "metadata": doc["metadata"],
                "doc_id": doc.get("doc_id"),
                "score": float(scores[0][i])
            })

        return results


class HybridRetriever:
    """Hybrid retriever with dense retrieval + reranking"""

    def __init__(self, config, reranker: Optional[BaseReranker] = None):
        """
        Initialize hybrid retriever

        Args:
            config: RAGConfig instance
            reranker: Optional reranker instance
        """
        self.config = config
        self.dense_retriever = DenseRetriever(config)

        # Set reranker
        if reranker is not None:
            self.reranker = reranker
        elif config.use_reranker:
            from .reranker import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(config.reranker_model)
        else:
            self.reranker = NoReranker()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_rerank: Optional[bool] = None
    ) -> List[Dict]:
        """
        Retrieve documents with optional reranking

        Args:
            query: Search query
            top_k: Number of final documents (default: config.top_k_rerank)
            use_rerank: Whether to use reranker (default: config.use_reranker)

        Returns:
            List of retrieved and optionally reranked documents
        """
        # Set defaults
        if top_k is None:
            top_k = self.config.top_k_rerank
        if use_rerank is None:
            use_rerank = self.config.use_reranker

        # Stage 1: Dense retrieval
        retrieval_k = self.config.top_k_retrieval if use_rerank else top_k
        candidates = self.dense_retriever.retrieve(query, top_k=retrieval_k)

        # Stage 2: Reranking (optional)
        if use_rerank and not isinstance(self.reranker, NoReranker):
            candidates = self.reranker.rerank(query, candidates, top_k=top_k)
        else:
            candidates = candidates[:top_k]

        return candidates

    def set_reranker(self, reranker: Optional[BaseReranker]):
        """Update reranker"""
        if reranker is None:
            self.reranker = NoReranker()
        else:
            self.reranker = reranker
