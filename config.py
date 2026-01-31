import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    with env_path.open() as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

load_env_file()
@dataclass
class Config:
    # API Keys
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")


    
    # Model configurations
    ASR_MODEL: str = "whisper-large-v3"  # Groq's free Whisper
    LLM_MODEL: str = "llama-3.1-70b-versatile"  # Groq's free Llama
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Local, free
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Local, free
    
    # Performance targets
    TARGET_TTFB_MS: int = 800
    MAX_CHUNK_SIZE: int = 256  # Smaller chunks for voice
    CHUNK_OVERLAP: int = 50
    
    # Search configuration
    VECTOR_WEIGHT: float = 0.7
    BM25_WEIGHT: float = 0.3
    TOP_K_INITIAL: int = 20
    TOP_K_RERANKED: int = 5
    
    # Voice optimization
    MAX_SENTENCE_WORDS: int = 15
    TARGET_GRADE_LEVEL: int = 8  # Flesch-Kincaid

config = Config()