import asyncio
from typing import AsyncGenerator, Callable, Optional
from groq import AsyncGroq
import numpy as np

class StreamingASR:
    """
    ASR with partial transcript emission for speculative RAG pre-fetching.
    Uses Groq's free Whisper API.
    """
    
    def __init__(self, api_key: str, on_partial: Optional[Callable] = None):
        self.client = AsyncGroq(api_key=api_key)
        self.on_partial = on_partial  # Callback for partial transcripts
        self.partial_buffer = ""
        self.confidence_threshold = 0.7
        
    async def transcribe_streaming(
        self, 
        audio_stream: AsyncGenerator[bytes, None]
    ) -> AsyncGenerator[dict, None]:
        """
        Process audio stream and emit partial transcripts for pre-fetching.
        
        Yields:
            {"type": "partial" | "final", "text": str, "confidence": float}
        """
        audio_buffer = b""
        chunk_duration_ms = 500  # Process every 500ms for low latency
        
        async for audio_chunk in audio_stream:
            audio_buffer += audio_chunk
            
            # Emit partial transcript when we have enough audio
            if len(audio_buffer) >= 16000 * (chunk_duration_ms / 1000) * 2:  # 16kHz, 16-bit
                partial = await self._process_chunk(audio_buffer)
                
                if partial and partial["confidence"] >= self.confidence_threshold:
                    yield {"type": "partial", **partial}
                    
                    # Trigger speculative pre-fetch
                    if self.on_partial:
                        asyncio.create_task(self.on_partial(partial["text"]))
        
        # Final transcription with full audio
        final = await self._transcribe_full(audio_buffer)
        yield {"type": "final", **final}
    
    async def _process_chunk(self, audio_data: bytes) -> Optional[dict]:
        """Process a chunk of audio for partial transcript."""
        try:
            # Convert bytes to audio file-like object
            audio_file = self._bytes_to_audio_file(audio_data)
            
            response = await self.client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                response_format="verbose_json"
            )
            
            return {
                "text": response.text,
                "confidence": getattr(response, 'confidence', 0.8)
            }
        except Exception as e:
            print(f"Partial transcription error: {e}")
            return None
    
    async def _transcribe_full(self, audio_data: bytes) -> dict:
        """Full transcription for final result."""
        audio_file = self._bytes_to_audio_file(audio_data)
        
        response = await self.client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=audio_file,
            response_format="verbose_json"
        )
        
        return {
            "text": response.text,
            "confidence": 1.0
        }
    
    def _bytes_to_audio_file(self, audio_data: bytes):
        """Convert raw bytes to a file-like object for the API."""
        import io
        return ("audio.wav", io.BytesIO(audio_data), "audio/wav")