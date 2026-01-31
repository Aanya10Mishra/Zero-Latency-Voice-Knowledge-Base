import time
import requests
import pyttsx3

def run_demo():
    """Demonstrate the Zero-Latency Voice RAG pipeline"""
    
    print("="*60)
    print("🎯 ZERO-LATENCY VOICE RAG DEMO")
    print("="*60)
    
    tts = pyttsx3.init()
    tts.setProperty('rate', 160)
    
    # Test queries
    queries = [
        "What is this manual about?",
        "How do I create a table?",
        "What are the main features?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i} ---")
        print(f"❓ Question: {query}")
        
        # Measure TTFB
        start = time.time()
        
        # Speak filler immediately
        tts.say("Let me check that for you.")
        filler_start = time.time()
        
        # Query API in parallel (simulated - actually sequential here)
        response = requests.post(
            "http://localhost:8000/query",
            params={"query": query, "conversation_id": "demo"}
        )
        
        tts.runAndWait()
        filler_time = (time.time() - filler_start) * 1000
        
        if response.status_code == 200:
            answer = response.json()["response"]
            total_time = (time.time() - start) * 1000
            
            print(f"✅ Answer: {answer[:100]}...")
            print(f"⏱️  Filler TTFB: {filler_time:.0f}ms | Total: {total_time:.0f}ms")
            
            # Speak answer
            tts.say(answer)
            tts.runAndWait()
        else:
            print(f"❌ Error: {response.status_code}")
        
        time.sleep(1)
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    run_demo()