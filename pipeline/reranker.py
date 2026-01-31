import asyncio
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from pipeline.hybrid_search import SearchResult

class CrossEncoderReranker:
    """
    Reranks search results using a cross-encoder model.
    Implements async execution to allow filler generation in parallel.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
    async def rerank(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Rerank results using cross-encoder.
        Returns top_k results sorted by relevance.
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        reranked = await loop.run_in_executor(
            None,
            self._rerank_sync,
            query,
            results,
            top_k
        )
        return reranked
    
    def _rerank_sync(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int
    ) -> List[SearchResult]:
        """Synchronous reranking logic."""
        if not results:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, r.content] for r in results]
        
        # Tokenize
        features = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)
        
        # Get scores
        with torch.no_grad():
            scores = self.model(**features).logits.squeeze(-1)
        
        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)[:top_k]
        
        reranked_results = []
        for idx in sorted_indices:
            result = results[idx]
            result.score = scores[idx].item()
            reranked_results.append(result)
        
        return reranked_results


class FillerAndRerank:
    """
    Coordinates filler generation and reranking in parallel.
    Ensures TTFB < 800ms by starting filler immediately.
    """
    
    def __init__(self, reranker: CrossEncoderReranker, llm_client):
        self.reranker = reranker
        self.llm_client = llm_client
        
        # Pre-defined fillers for variety
        self.fillers = [
            "Let me look that up for you...",
            "One moment while I check the manual...",
            "I'm finding the relevant information...",
            "Let me search the technical documentation...",
            "Checking the specifications now..."
        ]
        self.filler_index = 0
    
    async def process_with_filler(
        self,
        query: str,
        search_results: List[SearchResult],
        on_filler_ready: callable
    ) -> List[SearchResult]:
        """
        Start filler generation immediately, rerank in background.
        
        Args:
            query: User's query
            search_results: Initial search results
            on_filler_ready: Callback when filler audio should start
        
        Returns:
            Reranked results
        """
        # Start both tasks in parallel
        filler_task = asyncio.create_task(self._generate_filler(on_filler_ready))
        rerank_task = asyncio.create_task(
            self.reranker.rerank(query, search_results)
        )
        
        # Wait for both to complete
        await filler_task
        reranked = await rerank_task
        
        return reranked
    
    async def _generate_filler(self, on_ready: callable):
        """Generate and emit filler response immediately."""
        filler = self.fillers[self.filler_index]
        self.filler_index = (self.filler_index + 1) % len(self.fillers)
        
        # Notify immediately for TTS
        if on_ready:
            await on_ready(filler)