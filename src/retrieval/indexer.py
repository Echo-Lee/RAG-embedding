"""FAISS index builder for email documents"""
import json
import hashlib
from pathlib import Path
from typing import List, Optional
import numpy as np
import faiss
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


class FAISSIndexBuilder:
    """Build and save FAISS index for document retrieval"""

    def __init__(self, config):
        """
        Initialize index builder

        Args:
            config: RAGConfig instance
        """
        self.config = config
        self.index = None
        self.documents = None
        self.model = None

    def load_model(self):
        """Load embedding model"""
        if self.model is not None:
            return self.model

        print(f"Loading embedding model: {self.config.embedding_model}")

        # Check if using fine-tuned model
        if self.config.use_finetuned and self.config.finetuned_model_path:
            model_path = str(self.config.finetuned_model_path)
            print(f"Loading fine-tuned model from: {model_path}")
        else:
            model_path = self.config.embedding_model

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=device)
        self.model.max_seq_length = self.config.max_seq_length

        print(f"Model loaded on device: {device}")
        return self.model

    def build_index(self, documents: List) -> faiss.Index:
        """
        Build FAISS index from documents

        Args:
            documents: List of EmailDocument instances

        Returns:
            FAISS index
        """
        self.documents = documents

        if len(documents) == 0:
            raise ValueError("No documents to index")

        # Load model
        model = self.load_model()

        # Extract text content
        corpus_texts = [doc.content for doc in documents]

        print(f"Encoding {len(corpus_texts)} documents...")

        # Encode documents in batches
        embeddings = model.encode(
            corpus_texts,
            show_progress_bar=True,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        print(f"Embeddings shape: {embeddings.shape}")

        # Build FAISS index (Inner Product for normalized vectors = cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"Built FAISS index with {self.index.ntotal} vectors")

        return self.index

    def save_index(self):
        """Save index and metadata to disk"""
        if self.index is None or self.documents is None:
            raise ValueError("Index not built yet. Call build_index() first.")

        # Save FAISS index
        index_path = self.config.index_path
        faiss.write_index(self.index, str(index_path))
        print(f"Saved FAISS index to: {index_path}")

        # Save document metadata
        metadata_path = self.config.metadata_path
        metadata = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "doc_id": doc.doc_id
            }
            for doc in self.documents
        ]

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved metadata to: {metadata_path}")

        # Save config cache
        config_cache = {
            "embedding_model": self.config.embedding_model,
            "use_finetuned": self.config.use_finetuned,
            "num_documents": len(self.documents),
            "config_hash": self._compute_config_hash()
        }

        config_cache_path = self.config.config_cache_path
        with open(config_cache_path, 'w', encoding='utf-8') as f:
            json.dump(config_cache, f, indent=2)
        print(f"Saved config cache to: {config_cache_path}")

    def load_index(self) -> tuple:
        """
        Load existing index and metadata

        Returns:
            Tuple of (index, metadata_list)
        """
        index_path = self.config.index_path
        metadata_path = self.config.metadata_path

        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))
        print(f"Loaded FAISS index: {self.index.ntotal} vectors")

        # Load metadata
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {len(metadata)} documents")

        return self.index, metadata

    def index_exists(self) -> bool:
        """Check if index files exist"""
        return (
            self.config.index_path.exists() and
            self.config.metadata_path.exists()
        )

    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for cache validation"""
        config_str = (
            f"{self.config.embedding_model}|"
            f"{self.config.dataset.data_path}|"
            f"{len(self.documents) if self.documents else 0}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
