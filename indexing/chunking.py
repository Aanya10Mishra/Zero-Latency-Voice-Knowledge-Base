from typing import List, Dict
import re
from dataclasses import dataclass

@dataclass
class VoiceChunk:
    content: str
    metadata: Dict
    word_count: int
    estimated_speech_duration_sec: float

class VoiceOptimizedChunker:
    """
    Creates chunks optimized for voice synthesis.
    - Shorter sentences
    - Natural pause points
    - Technical term handling
    """
    
    def __init__(
        self,
        max_chunk_words: int = 50,  # ~15 seconds of speech
        overlap_words: int = 10,
        words_per_minute: int = 150  # Average speech rate
    ):
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
        self.wpm = words_per_minute
    
    def chunk_document(
        self,
        text: str,
        metadata: Dict = None
    ) -> List[VoiceChunk]:
        """
        Chunk document with voice-first approach.
        Preserves semantic boundaries and natural speech breaks.
        """
        # Pre-process: add pronunciation guides for technical terms
        processed_text = self._add_pronunciation_guides(text)
        
        # Split into sentences first
        sentences = self._split_into_sentences(processed_text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = len(sentence.split())
            
            # If single sentence exceeds max, split it
            if sentence_words > self.max_chunk_words:
                # Save current chunk if exists
                if current_chunk:
                    chunks.append(self._create_chunk(current_chunk, metadata))
                    current_chunk = []
                    current_word_count = 0
                
                # Split long sentence at natural pauses
                sub_sentences = self._split_long_sentence(sentence)
                for sub in sub_sentences:
                    chunks.append(self._create_chunk([sub], metadata))
            
            # Normal case: add sentence to current chunk
            elif current_word_count + sentence_words <= self.max_chunk_words:
                current_chunk.append(sentence)
                current_word_count += sentence_words
            
            # Chunk is full, start new one with overlap
            else:
                chunks.append(self._create_chunk(current_chunk, metadata))
                
                # Add overlap from end of previous chunk
                overlap = self._get_overlap(current_chunk)
                current_chunk = overlap + [sentence]
                current_word_count = sum(len(s.split()) for s in current_chunk)
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(self._create_chunk(current_chunk, metadata))
        
        return chunks
    
    def _add_pronunciation_guides(self, text: str) -> str:
        """Add phonetic hints for technical terms."""
        # Common technical term pronunciations
        pronunciations = {
            r'\bGPU\b': 'G P U',
            r'\bCPU\b': 'C P U',
            r'\bRAM\b': 'RAM',
            r'\bSSD\b': 'S S D',
            r'\bNVMe\b': 'N V M E',
            r'\bPCIe\b': 'P C I E',
            r'\bUSB\b': 'U S B',
            r'\bHDMI\b': 'H D M I',
            r'\bAPI\b': 'A P I',
            r'\bHTTP\b': 'H T T P',
            r'\bJSON\b': 'Jason',
            r'\bSQL\b': 'sequel',
            r'\bGHz\b': 'gigahertz',
            r'\bMHz\b': 'megahertz',
            r'\bTB\b': 'terabytes',
            r'\bGB\b': 'gigabytes',
            r'\bMB\b': 'megabytes',
        }
        
        result = text
        for pattern, replacement in pronunciations.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences, preserving technical notation."""
        # Handle common abbreviations and technical notation
        text = re.sub(r'(?<=[A-Z])\.(?=[A-Z])', '<DOT>', text)  # Abbreviations
        text = re.sub(r'(\d+)\.(\d+)', r'\1<DECIMAL>\2', text)  # Decimal numbers
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Restore
        sentences = [
            s.replace('<DOT>', '.').replace('<DECIMAL>', '.')
            for s in sentences
        ]
        
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str) -> List[str]:
        """Split long sentence at natural pause points."""
        # Split at commas, semicolons, colons, "and", "or"
        parts = re.split(r'(?<=[,;:])\s+|(?:\s+(?:and|or)\s+)', sentence)
        
        result = []
        current = []
        current_words = 0
        
        for part in parts:
            part_words = len(part.split())
            if current_words + part_words <= self.max_chunk_words:
                current.append(part)
                current_words += part_words
            else:
                if current:
                    result.append(' '.join(current))
                current = [part]
                current_words = part_words
        
        if current:
            result.append(' '.join(current))
        
        return result
    
    def _get_overlap(self, sentences: List[str]) -> List[str]:
        """Get last N words worth of sentences for overlap."""
        if not sentences:
            return []
        
        overlap = []
        word_count = 0
        
        for sentence in reversed(sentences):
            sentence_words = len(sentence.split())
            if word_count + sentence_words <= self.overlap_words:
                overlap.insert(0, sentence)
                word_count += sentence_words
            else:
                break
        
        return overlap
    
    def _create_chunk(self, sentences: List[str], metadata: Dict) -> VoiceChunk:
        content = ' '.join(sentences)
        word_count = len(content.split())
        duration = (word_count / self.wpm) * 60
        
        return VoiceChunk(
            content=content,
            metadata=metadata or {},
            word_count=word_count,
            estimated_speech_duration_sec=duration
        )