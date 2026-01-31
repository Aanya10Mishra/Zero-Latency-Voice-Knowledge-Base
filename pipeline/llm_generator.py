import asyncio
import re
from typing import AsyncGenerator, List
from groq import AsyncGroq
from pipeline.hybrid_search import SearchResult
from pipeline.voice_optimizer import VoiceOptimizer

class StreamingLLMGenerator:
    """
    Generates streaming responses optimized for voice output.
    """
    
    def __init__(self, api_key: str):
        self.client = AsyncGroq(api_key=api_key)
        self.voice_optimizer = VoiceOptimizer(api_key)
        
        self.system_prompt = """You are a helpful voice assistant for technical support. 
Your responses will be read aloud, so:
- Keep sentences short and clear (max 15 words each)
- Use simple, conversational language
- Avoid jargon; explain technical terms if needed
- Be direct and actionable
- Structure information in easy-to-follow steps"""

    async def generate_stream(
        self,
        query: str,
        context_docs: List[SearchResult],
        conversation_history: List[dict] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream LLM response in voice-optimized chunks.
        """
        # Build context from documents
        context = self._build_context(context_docs)
        
        # Build messages
        messages = [{"role": "system", "content": self.system_prompt}]
        
        if conversation_history:
            for turn in conversation_history[-3:]:
                messages.append({"role": "user", "content": turn['user']})
                messages.append({"role": "assistant", "content": turn['assistant']})
        
        messages.append({
            "role": "user",
            "content": f"""Based on the following technical documentation, answer the user's question.

Documentation:
{context}

User Question: {query}

Provide a clear, spoken-style answer:"""
        })
        
        try:
            # NON-STREAMING first (more reliable)
            response = await self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Faster model
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            
            full_response = response.choices[0].message.content
            
            # Split into sentences and yield
            sentences = re.split(r'(?<=[.!?])\s+', full_response)
            
            for sentence in sentences:
                if sentence.strip():
                    # Quick optimization
                    optimized = self._quick_optimize(sentence)
                    yield optimized
                    
        except Exception as e:
            print(f"LLM Error: {e}")
            yield "I'm sorry, I encountered an error. Please try again."
    
    def _build_context(self, docs: List[SearchResult]) -> str:
        """Build context string from search results."""
        if not docs:
            return "No relevant documentation found."
        
        context_parts = []
        for i, doc in enumerate(docs[:5], 1):
            source = doc.metadata.get('source', 'Unknown')
            page = doc.metadata.get('page', '')
            context_parts.append(f"[Source {i}: {source} Page {page}]\n{doc.content}")
        
        return "\n\n".join(context_parts)
    
    def _quick_optimize(self, text: str) -> str:
        """Quick text optimization for voice."""
        # Remove markdown
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        text = re.sub(r'#+\s*', '', text)
        
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()