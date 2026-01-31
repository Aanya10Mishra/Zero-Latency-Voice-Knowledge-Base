import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import config
from pipeline.query_rewriter import QueryRewriter
from pipeline.hybrid_search import HybridSearch
from pipeline.reranker import CrossEncoderReranker
from pipeline.llm_generator import StreamingLLMGenerator

async def test():
    print("1. Testing config...")
    print(f"   GROQ_API_KEY set: {bool(config.GROQ_API_KEY)}")
    print(f"   Key starts with: {config.GROQ_API_KEY[:10]}..." if config.GROQ_API_KEY else "   NO KEY FOUND!")
    
    print("\n2. Testing HybridSearch...")
    try:
        search = HybridSearch()
        print("   ✓ HybridSearch initialized")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    print("\n3. Testing QueryRewriter...")
    try:
        rewriter = QueryRewriter(config.GROQ_API_KEY)
        print("   ✓ QueryRewriter initialized")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    print("\n4. Testing search...")
    try:
        results = await search.search("hello", top_k=5)
        print(f"   ✓ Search returned {len(results)} results")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    print("\n5. Testing LLM...")
    try:
        llm = StreamingLLMGenerator(config.GROQ_API_KEY)
        print("   ✓ LLM initialized")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    print("\n✅ All components working!")

if __name__ == "__main__":
    asyncio.run(test())
