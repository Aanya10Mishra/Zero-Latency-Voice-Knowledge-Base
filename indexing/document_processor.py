import asyncio
from pathlib import Path
from typing import List
from PyPDF2 import PdfReader
from indexing.chunking import VoiceOptimizedChunker, VoiceChunk
from pipeline.hybrid_search import HybridSearch

from .chunking import VoiceOptimizedChunker, VoiceChunk
import sys
sys.path.append('..')

class DocumentProcessor:
    """
    Processes technical manuals and indexes them for RAG.
    Supports PDF and text formats.
    """
    
    def __init__(self):
        self.chunker = VoiceOptimizedChunker()
        self.search = HybridSearch()
    
    async def ingest_pdf(self, pdf_path: str):
        """
        Ingest a PDF technical manual.
        """
        print(f"Processing: {pdf_path}")
        reader = PdfReader(pdf_path)
        
        all_chunks: List[VoiceChunk] = []
        
        for page_num, page in enumerate(reader.pages, 1):
            text = page.extract_text()
            if not text.strip():
                continue
            
            metadata = {
                "source": Path(pdf_path).name,
                "page": page_num
            }
            
            chunks = self.chunker.chunk_document(text, metadata)
            all_chunks.extend(chunks)
            
            print(f"  Page {page_num}: {len(chunks)} chunks")
        
        # Index chunks
        await self._index_chunks(all_chunks)
        print(f"Indexed {len(all_chunks)} total chunks")
    
    async def _index_chunks(self, chunks: List[VoiceChunk]):
        """Add chunks to the vector store."""
        documents = []
        metadatas = []
        ids = []
        
        for i, chunk in enumerate(chunks):
            documents.append(chunk.content)
            metadatas.append({
                **chunk.metadata,
                "word_count": chunk.word_count,
                "duration_sec": chunk.estimated_speech_duration_sec
            })
            ids.append(f"chunk_{i}")
        
        # Compute embeddings
        embeddings = self.search.embedder.encode(documents).tolist()
        
        # Add to ChromaDB
        self.search.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )


async def main():
    """Ingest sample technical manual."""
    processor = DocumentProcessor()
    
    # Example: Ingest a hardware manual
    # You can use any technical PDF, such as:
    # - ThinkPad Hardware Maintenance Manual
    # - Arduino Reference Manual
    # - Raspberry Pi Documentation
    
    await processor.ingest_pdf("./data/technical_manual.pdf")


if __name__ == "__main__":
    asyncio.run(main())