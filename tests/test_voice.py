import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pyttsx3
import requests

def test_voice_output():
    """Test text-to-speech output"""
    print("Testing TTS...")
    
    engine = pyttsx3.init()
    engine.setProperty('rate', 175)
    
    # List available voices
    voices = engine.getProperty('voices')
    print(f"Available voices: {len(voices)}")
    for i, voice in enumerate(voices):
        print(f"  [{i}] {voice.name}")
    
    # Test speech
    engine.say("Hello! The voice system is working correctly.")
    engine.runAndWait()
    print("✓ TTS working!\n")

def test_full_pipeline():
    """Test the complete RAG pipeline with voice output"""
    print("Testing full pipeline with voice...")
    
    # Initialize TTS
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    
    # Query the RAG API
    query = "How do I create a table?"
    print(f"Query: {query}")
    
    response = requests.post(
        "http://localhost:8000/query",
        params={"query": query, "conversation_id": "voice_test"}
    )
    
    if response.status_code == 200:
        data = response.json()
        answer = data["response"]
        
        print(f"\nResponse: {answer}\n")
        print("Speaking response...")
        
        # Speak the response
        engine.say(answer)
        engine.runAndWait()
        
        print("✓ Full pipeline working!")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    print("="*50)
    print("VOICE RAG PIPELINE TEST")
    print("="*50 + "\n")
    
    # Test TTS first
    test_voice_output()
    
    # Test full pipeline
    input("Press Enter to test full pipeline (make sure server is running)...")
    test_full_pipeline()
