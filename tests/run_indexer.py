import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from PyPDF2 import PdfReader
from pipeline.hybrid_search import HybridSearch
from indexing.chunking import VoiceOptimizedChunker

async def main():
    print("Starting document indexing...")
    
    # Use LARGER chunks to reduce count
    chunker = VoiceOptimizedChunker(
        max_chunk_words=200,  # Even larger chunks
        overlap_words=20
    )
    search = HybridSearch()
    
    # Clear existing data first
    print("Clearing old data...")
    try:
        search.chroma_client.delete_collection("technical_manual")
        search.collection = search.chroma_client.get_or_create_collection(
            name="technical_manual",
            metadata={"hnsw:space": "cosine"}
        )
    except:
        pass
    
    pdf_path = "C:\\Users\\Manvi\\Documents\\Zero Latency-Voice Knowledge Base\\data\\postgresql-17-A4_compressed.pdf"
    print(f"Reading: {pdf_path}")
    
    try:
        reader = PdfReader(pdf_path)
        print(f"Found {len(reader.pages)} pages")
    except FileNotFoundError:
        print(f"ERROR: File not found at {pdf_path}")
        return
    
    all_chunks = []
    
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        if not text or not text.strip():
            continue
        
        metadata = {
            "source": Path(pdf_path).name,
            "page": str(page_num)
        }
        
        chunks = chunker.chunk_document(text, metadata)
        all_chunks.extend(chunks)
        
        if page_num % 100 == 0:
            print(f"  Processed {page_num} pages... ({len(all_chunks)} chunks so far)")
    
    print(f"\nTotal chunks: {len(all_chunks)}")
    
    # Limit chunks for faster testing
    MAX_CHUNKS = 3000
    if len(all_chunks) > MAX_CHUNKS:
        print(f"Limiting to first {MAX_CHUNKS} chunks for faster indexing...")
        all_chunks = all_chunks[:MAX_CHUNKS]
    
    print("Indexing to vector database...")
    
    # Use SMALL batches (ChromaDB max is 5461, we use 100 for safety)
    BATCH_SIZE = 100
    documents = [chunk.content for chunk in all_chunks]
    metadatas = [chunk.metadata for chunk in all_chunks]
    
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]
        batch_meta = metadatas[i:i+BATCH_SIZE]
        batch_ids = [f"chunk_{j}" for j in range(i, i+len(batch_docs))]
        
        # Create embeddings for this batch
        embeddings = search.embedder.encode(batch_docs).tolist()
        
        # Add to database
        search.collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            metadatas=batch_meta,
            ids=batch_ids
        )
        
        batch_num = (i // BATCH_SIZE) + 1
        print(f"  Batch {batch_num}/{total_batches} done ({min(i+BATCH_SIZE, len(documents))}/{len(documents)} chunks)")
    
    print(f"\n✅ Successfully indexed {len(all_chunks)} chunks!")
    
    # Test search
    print("\nTesting search...")
    results = await search.search("introduction", top_k=3)
    print(f"Search test returned {len(results)} results")
    
    if results:
        print(f"First result: {results[0].content[:150]}...")

if __name__ == "__main__":
    asyncio.run(main())
