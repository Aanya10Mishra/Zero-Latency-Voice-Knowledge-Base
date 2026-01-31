import asyncio
import time
import re
from typing import AsyncGenerator
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import config
from pipeline.query_rewriter import QueryRewriter
from pipeline.hybrid_search import HybridSearch
from pipeline.reranker import CrossEncoderReranker
from pipeline.llm_generator import StreamingLLMGenerator
from pipeline.tts import StreamingTTS

app = FastAPI(title="Zero-Latency Voice RAG")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize components
print("Initializing components...")
search = HybridSearch()
reranker = CrossEncoderReranker()
query_rewriter = QueryRewriter(config.GROQ_API_KEY)
llm_generator = StreamingLLMGenerator(config.GROQ_API_KEY)
tts = StreamingTTS()
print("✓ All components initialized")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/query")
async def text_query(query: str, conversation_id: str = "default"):
    """
    Text-based query endpoint.
    """
    try:
        print(f"\n--- New Query ---")
        print(f"Query: {query}")
        
        # Step 1: Rewrite query
        print("1. Rewriting query...")
        rewrite_result = await query_rewriter.rewrite_query(query, conversation_id)
        rewritten_query = rewrite_result["rewritten"]
        print(f"   Rewritten: {rewritten_query}")
        
        # Step 2: Search
        print("2. Searching...")
        search_results = await search.search(rewritten_query, top_k=10)
        print(f"   Found {len(search_results)} results")
        
        # Step 3: Rerank (skip if no results)
        if search_results:
            print("3. Reranking...")
            reranked = await reranker.rerank(rewritten_query, search_results, top_k=5)
            print(f"   Reranked to {len(reranked)} results")
        else:
            reranked = []
            print("3. Skipping rerank (no results)")
        
        # Step 4: Generate response
        print("4. Generating response...")
        response_chunks = []
        async for chunk in llm_generator.generate_stream(
            rewritten_query,
            reranked,
            query_rewriter.memory.get_history(conversation_id)
        ):
            response_chunks.append(chunk)
        
        response = " ".join(response_chunks)
        print(f"   Response: {response[:100]}...")
        
        # Update history
        query_rewriter.update_history(conversation_id, query, response)
        
        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "response": response,
            "sources": [{"page": r.metadata.get("page"), "score": r.score} for r in reranked[:3]]
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)