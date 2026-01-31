import asyncio
from typing import List, Optional
from groq import AsyncGroq
from utils.conversation_memory import ConversationMemory

class QueryRewriter:
    """
    Resolves ambiguous references using conversation history.
    Handles "the second one", "that thing", "it", etc.
    """
    
    def __init__(self, api_key: str):
        self.client = AsyncGroq(api_key=api_key)
        self.memory = ConversationMemory(max_turns=10)
        
    async def rewrite_query(
        self, 
        query: str, 
        conversation_id: str
    ) -> dict:
        """
        Rewrite query to resolve references and add context.
        
        Returns:
            {
                "original": str,
                "rewritten": str,
                "entities_resolved": List[str],
                "requires_context": bool
            }
        """
        history = self.memory.get_history(conversation_id)
        
        # Quick check if query needs rewriting
        needs_rewrite = self._needs_context_resolution(query)
        
        if not needs_rewrite:
            return {
                "original": query,
                "rewritten": query,
                "entities_resolved": [],
                "requires_context": False
            }
        
        # Use LLM to resolve references
        prompt = self._build_rewrite_prompt(query, history)
        
        response = await self.client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast model for quick rewrites
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        
        rewritten = response.choices[0].message.content.strip()
        
        return {
            "original": query,
            "rewritten": rewritten,
            "entities_resolved": self._extract_resolved_entities(query, rewritten),
            "requires_context": True
        }
    
    def _needs_context_resolution(self, query: str) -> bool:
        """Quick heuristic check for ambiguous references."""
        ambiguous_patterns = [
            "the first", "the second", "the third", "the last",
            "that one", "this one", "the other",
            "it", "them", "those", "these",
            "what about", "and the", "also the",
            "same thing", "previous", "earlier"
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in ambiguous_patterns)
    
    def _build_rewrite_prompt(self, query: str, history: List[dict]) -> str:
        history_text = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in history[-5:]  # Last 5 turns
        ])
        
        return f"""Given the conversation history and the current query, rewrite the query to be self-contained by resolving all references.

Conversation History:
{history_text}

Current Query: "{query}"

Rules:
1. Replace pronouns (it, they, that) with the specific entity
2. Replace ordinal references (the second one, the first) with the actual item
3. Add any implicit context needed
4. Keep the rewritten query concise but complete

Rewritten Query (only output the rewritten query, nothing else):"""

    def _extract_resolved_entities(self, original: str, rewritten: str) -> List[str]:
        """Extract what entities were resolved in the rewrite."""
        # Simple diff to find added terms
        original_words = set(original.lower().split())
        rewritten_words = set(rewritten.lower().split())
        return list(rewritten_words - original_words)
    
    def update_history(self, conversation_id: str, user: str, assistant: str):
        """Update conversation history after a turn."""
        self.memory.add_turn(conversation_id, user, assistant)