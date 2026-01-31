import asyncio
from typing import List, Tuple
from dataclasses import dataclass
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
import math
from collections import Counter

@dataclass
class SearchResult:
    content: str
    metadata: dict
    score: float
    source: str  # "vector" or "bm25" or "hybrid"

class BM25Okapi:
    """Lightweight BM25 implementation to avoid external dependencies."""
    def __init__(
        self,
        corpus: List[List[str]],
        k1: float = 1.5,
        b: float = 0.75
    ):
        self.corpus = corpus
        self.corpus_size = len(corpus)
        self.avg_doc_len = (
            sum(len(doc) for doc in corpus) / self.corpus_size
            if self.corpus_size else 0.0
        )
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.term_doc_freq = {}

        for doc in corpus:
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            for term in freq:
                self.term_doc_freq[term] = self.term_doc_freq.get(term, 0) + 1

    def get_scores(self, query_terms: List[str]) -> np.ndarray:
        if self.corpus_size == 0:
            return np.zeros(0, dtype=float)

        scores = np.zeros(self.corpus_size, dtype=float)

        for term in query_terms:
            df = self.term_doc_freq.get(term, 0)
            if df == 0:
                continue

            idf = math.log(1 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for idx, freq in enumerate(self.doc_freqs):
                term_freq = freq.get(term, 0)
                if term_freq == 0:
                    continue

                doc_len = len(self.corpus[idx])
                denom = (
                    term_freq
                    + self.k1
                    * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                    if self.avg_doc_len
                    else term_freq + self.k1
                )
                scores[idx] += idf * (term_freq * (self.k1 + 1)) / denom

        return scores

class HybridSearch:
    """
    Combines vector similarity search with BM25 keyword matching.
    Implements Reciprocal Rank Fusion for score combination.
    """
    
    def __init__(
        self,
        collection_name: str = "technical_manual",
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ):
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        
        # Initialize vector store
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model (local, free)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # BM25 index (built from documents)
        self.bm25_index = None
        self.documents = []
        
    async def search(
        self, 
        query: str, 
        top_k: int = 20
    ) -> List[SearchResult]:
        """
        Perform hybrid search combining vector and BM25 results.
        Uses asyncio for parallel execution.
        """
        # Run both searches in parallel
        vector_task = asyncio.create_task(self._vector_search(query, top_k))
        bm25_task = asyncio.create_task(self._bm25_search(query, top_k))
        
        vector_results, bm25_results = await asyncio.gather(
            vector_task, bm25_task
        )
        
        # Combine with Reciprocal Rank Fusion
        combined = self._reciprocal_rank_fusion(
            vector_results, 
            bm25_results,
            top_k
        )
        
        return combined
    
    async def _vector_search(
        self, 
        query: str, 
        top_k: int
    ) -> List[Tuple[str, float, dict]]:
        """Vector similarity search using ChromaDB."""
        query_embedding = self.embedder.encode(query).tolist()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        return [
            (doc, 1 - dist, meta)  # Convert distance to similarity
            for doc, dist, meta in zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )
        ]
    
    async def _bm25_search(
        self, 
        query: str, 
        top_k: int
    ) -> List[Tuple[str, float, dict]]:
        """BM25 keyword search."""
        if self.bm25_index is None:
            await self._build_bm25_index()
        
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = self.documents[idx]
                results.append((doc['content'], scores[idx], doc['metadata']))
        
        return results
    
    async def _build_bm25_index(self):
        """Build BM25 index from ChromaDB documents."""
        # Fetch all documents from ChromaDB
        all_docs = self.collection.get(include=["documents", "metadatas"])
        
        self.documents = [
            {"content": doc, "metadata": meta}
            for doc, meta in zip(all_docs['documents'], all_docs['metadatas'])
        ]
        
        tokenized_docs = [doc['content'].lower().split() for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_docs)
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple],
        bm25_results: List[Tuple],
        top_k: int,
        k: int = 60  # RRF constant
    ) -> List[SearchResult]:
        """
        Combine results using Reciprocal Rank Fusion.
        RRF score = sum(1 / (k + rank)) for each result list
        """
        scores = {}
        content_map = {}
        metadata_map = {}
        
        # Score vector results
        for rank, (content, score, metadata) in enumerate(vector_results):
            content_key = hash(content)
            rrf_score = self.vector_weight * (1 / (k + rank + 1))
            scores[content_key] = scores.get(content_key, 0) + rrf_score
            content_map[content_key] = content
            metadata_map[content_key] = metadata
        
        # Score BM25 results
        for rank, (content, score, metadata) in enumerate(bm25_results):
            content_key = hash(content)
            rrf_score = self.bm25_weight * (1 / (k + rank + 1))
            scores[content_key] = scores.get(content_key, 0) + rrf_score
            content_map[content_key] = content
            metadata_map[content_key] = metadata
        
        # Sort by combined score
        sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        return [
            SearchResult(
                content=content_map[key],
                metadata=metadata_map[key],
                score=scores[key],
                source="hybrid"
            )
            for key in sorted_keys[:top_k]
        ]