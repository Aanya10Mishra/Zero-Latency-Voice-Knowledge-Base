import asyncio
import pyttsx3
import io
import wave
import tempfile
import os
from typing import AsyncGenerator

class StreamingTTS:
    """
    Text-to-Speech using pyttsx3.
    Fully offline, works with all Python versions.
    """
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self._configure_engine()
    
    def _configure_engine(self):
        """Configure TTS engine for natural speech."""
        # Set speech rate (words per minute)
        # Default is ~200, lower for clearer speech
        self.engine.setProperty('rate', 175)
        
        # Set volume (0.0 to 1.0)
        self.engine.setProperty('volume', 0.9)
        
        # Select voice (optional - use system default if not set)
        voices = self.engine.getProperty('voices')
        if voices:
            # Try to find a natural-sounding voice
            # Index 0 is usually male, 1 is usually female
            for voice in voices:
                if 'english' in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    break
    
    async def synthesize_stream(
        self,
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[bytes, None]:
        """
        Convert streaming text to streaming audio.
        
        Yields:
            Audio bytes as they become available
        """
        async for text_chunk in text_stream:
            if not text_chunk.strip():
                continue
            
            audio = await self._synthesize_chunk(text_chunk)
            if audio:
                yield audio
    
    async def _synthesize_chunk(self, text: str) -> bytes:
        """Synthesize a single text chunk to audio bytes."""
        # Run pyttsx3 in thread pool (it's synchronous)
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            None,
            self._synthesize_sync,
            text
        )
        return audio_bytes
    
    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous synthesis to a temporary file, then read bytes."""
        # Create temp file for audio output
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Save speech to file
            self.engine.save_to_file(text, tmp_path)
            self.engine.runAndWait()
            
            # Read the audio bytes
            with open(tmp_path, 'rb') as f:
                audio_bytes = f.read()
            
            return audio_bytes
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    
    def speak_direct(self, text: str):
        """
        Speak text directly through speakers.
        Use this for lowest latency when streaming to file isn't needed.
        """
        self.engine.say(text)
        self.engine.runAndWait()


class LowLatencyTTS:
    """
    Alternative TTS class optimized for TTFB.
    Speaks directly instead of saving to file for faster response.
    """
    
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 175)
        self.engine.setProperty('volume', 0.9)
        self._select_best_voice()
    
    def _select_best_voice(self):
        """Select the most natural-sounding voice available."""
        voices = self.engine.getProperty('voices')
        
        # Priority order for voice selection
        preferred_keywords = ['david', 'zira', 'mark', 'english', 'us']
        
        for keyword in preferred_keywords:
            for voice in voices:
                if keyword in voice.name.lower():
                    self.engine.setProperty('voice', voice.id)
                    print(f"Selected voice: {voice.name}")
                    return
        
        # Fallback to first voice
        if voices:
            self.engine.setProperty('voice', voices[0].id)
    
    async def speak_streaming(
        self,
        text_stream: AsyncGenerator[str, None]
    ):
        """
        Speak text chunks as they arrive.
        Lowest latency - audio plays directly through speakers.
        """
        loop = asyncio.get_event_loop()
        
        async for text_chunk in text_stream:
            if not text_chunk.strip():
                continue
            
            # Speak in background thread
            await loop.run_in_executor(
                None,
                self._speak_sync,
                text_chunk
            )
    
    def _speak_sync(self, text: str):
        """Synchronous direct speech."""
        self.engine.say(text)
        self.engine.runAndWait()
    
    def list_available_voices(self):
        """Print all available voices on the system."""
        voices = self.engine.getProperty('voices')
        print("Available voices:")
        for i, voice in enumerate(voices):
            print(f"  [{i}] {voice.name} ({voice.id})")