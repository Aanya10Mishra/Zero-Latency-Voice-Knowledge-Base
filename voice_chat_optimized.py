import pyttsx3
import requests
import speech_recognition as sr
import threading
import time
import queue

class ZeroLatencyVoiceChat:
    """
    Optimized voice chat with:
    - TTFB measurement
    - Filler responses while processing
    - INTERRUPTIBLE speech (can say 'quit' anytime!)
    """
    
    def __init__(self):
        # Initialize TTS
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 160)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        self.conversation_id = "voice_session_1"
        self.api_url = "http://localhost:8000/query"
        
        # Filler responses
        self.fillers = [
            "Let me look that up for you.",
            "One moment please.",
            "Checking the manual now.",
            "Let me find that information.",
            "Searching the documentation."
        ]
        self.filler_index = 0
        
        # Control flags
        self.is_speaking = False
        self.should_stop = False
        self.interrupt_requested = False
    
    def speak(self, text):
        """Convert text to speech (can be interrupted)"""
        print(f"🔊 Assistant: {text}")
        self.is_speaking = True
        
        # Split into sentences for interruptibility
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if interrupted before speaking each sentence
            if self.interrupt_requested or self.should_stop:
                print("🛑 Speech interrupted!")
                self.tts.stop()
                break
            
            self.tts.say(sentence)
            self.tts.runAndWait()
        
        self.is_speaking = False
    
    def speak_non_blocking(self, text):
        """Speak in background thread (allows interruption)"""
        thread = threading.Thread(target=self.speak, args=(text,))
        thread.daemon = True
        thread.start()
        return thread
    
    def stop_speaking(self):
        """Stop any ongoing speech immediately"""
        self.interrupt_requested = True
        self.tts.stop()
        time.sleep(0.1)
        self.interrupt_requested = False
    
    def get_filler(self):
        """Get next filler response"""
        filler = self.fillers[self.filler_index]
        self.filler_index = (self.filler_index + 1) % len(self.fillers)
        return filler
    
    def listen(self, timeout=5):
        """Listen for voice input"""
        print("🎤 Listening... (speak now)")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                return None
        
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
    
    def listen_in_background(self):
        """Listen for 'quit' command while speaking"""
        result = {"text": None}
        
        def background_listen():
            try:
                with self.microphone as source:
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.2)
                    audio = self.recognizer.listen(source, timeout=2, phrase_time_limit=3)
                    text = self.recognizer.recognize_google(audio)
                    result["text"] = text.lower()
            except:
                pass
        
        thread = threading.Thread(target=background_listen)
        thread.daemon = True
        thread.start()
        return result
    
    def is_quit_command(self, text):
        """Check if user wants to quit"""
        if text is None:
            return False
        quit_words = ['quit', 'exit', 'bye', 'stop', 'goodbye', 'close', 'end']
        return text.lower().strip() in quit_words
    
    def query_rag(self, question):
        """Send question to RAG API"""
        try:
            response = requests.post(
                self.api_url,
                params={"query": question, "conversation_id": self.conversation_id}
            )
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return "Sorry, I encountered an error processing your question."
        except Exception as e:
            return f"Sorry, I couldn't connect to the server."
    
    def process_with_filler(self, question):
        """Process question with filler response for low TTFB"""
        result = {"answer": None}
        
        def fetch_answer():
            result["answer"] = self.query_rag(question)
        
        # Start RAG query in background
        rag_thread = threading.Thread(target=fetch_answer)
        rag_thread.start()
        
        # Speak filler immediately
        filler = self.get_filler()
        self.speak(filler)
        
        # Wait for RAG to finish
        rag_thread.join()
        
        return result["answer"]
    
    def run(self):
        """Main chat loop with interrupt support"""
        print("\n" + "="*50)
        print("⚡ ZERO-LATENCY VOICE ASSISTANT")
        print("="*50)
        print("\n💡 TIP: Say 'STOP' anytime to interrupt!")
        print("💡 TIP: Say 'QUIT' to exit the program\n")
        
        self.speak("Hello! I'm your technical assistant. Ask me anything. Say stop to interrupt, or quit to exit.")
        
        while not self.should_stop:
            # Listen for input
            user_input = self.listen()
            
            if user_input is None:
                self.speak("I didn't catch that. Please try again.")
                continue
            
            # Check for quit
            if self.is_quit_command(user_input):
                self.stop_speaking()
                self.speak("Goodbye!")
                print("\n👋 Session ended.")
                break
            
            # Check for stop/interrupt (skip this turn)
            if user_input.lower().strip() in ['stop', 'cancel', 'nevermind', 'never mind']:
                self.stop_speaking()
                self.speak("Okay, cancelled.")
                continue
            
            # Process with TTFB measurement
            start_time = time.time()
            
            # Get answer (filler plays immediately)
            answer = self.process_with_filler(user_input)
            
            # Speak answer in background (interruptible)
            speech_thread = self.speak_non_blocking(answer)
            
            # While speaking, listen for interrupt commands
            while speech_thread.is_alive():
                # Check for interrupt every 0.5 seconds
                bg_result = self.listen_in_background()
                time.sleep(0.5)
                
                if bg_result["text"]:
                    if self.is_quit_command(bg_result["text"]):
                        self.stop_speaking()
                        self.should_stop = True
                        break
                    elif bg_result["text"] in ['stop', 'cancel']:
                        self.stop_speaking()
                        self.speak("Stopped.")
                        break
            
            if self.should_stop:
                self.speak("Goodbye!")
                print("\n👋 Session ended.")
                break
            
            total_time = (time.time() - start_time) * 1000
            print(f"⏱️  Total response time: {total_time:.0f}ms")


class SimpleInterruptibleChat:
    """
    Simpler version - uses keyboard interrupt instead of voice.
    More reliable for interruption.
    """
    
    def __init__(self):
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 160)
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.api_url = "http://localhost:8000/query"
        self.conversation_id = "session_1"
    
    def speak(self, text):
        """Speak text"""
        print(f"🔊 Assistant: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
    def listen(self):
        """Listen for voice input"""
        print("\n🎤 Listening... (speak now, or say 'quit' to exit)")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
            try:
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            except sr.WaitTimeoutError:
                return None
        
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
        except sr.UnknownValueError:
            return None
        except:
            return None
    
    def query_rag(self, question):
        """Query the RAG API"""
        try:
            response = requests.post(
                self.api_url,
                params={"query": question, "conversation_id": self.conversation_id}
            )
            if response.status_code == 200:
                return response.json()["response"]
            return "Sorry, there was an error."
        except:
            return "Sorry, couldn't connect to server."
    
    def run(self):
        """Main loop"""
        print("\n" + "="*50)
        print("🎙️  VOICE ASSISTANT")
        print("="*50)
        print("\n• Speak your question after the beep")
        print("• Say 'QUIT' or 'EXIT' to stop")
        print("• Press Ctrl+C to force quit\n")
        
        self.speak("Hello! Ask me anything about the manual. Say quit to exit.")
        
        while True:
            try:
                user_input = self.listen()
                
                if user_input is None:
                    self.speak("I didn't hear anything. Try again.")
                    continue
                
                # Check for quit
                if user_input.lower().strip() in ['quit', 'exit', 'bye', 'goodbye', 'stop']:
                    self.speak("Goodbye!")
                    break
                
                # Get and speak answer
                self.speak("Looking that up...")
                answer = self.query_rag(user_input)
                self.speak(answer)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break


if __name__ == "__main__":
    print("\nChoose mode:")
    print("1. Simple Voice Chat (more reliable)")
    print("2. Advanced Interruptible Chat")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    if choice == "2":
        chat = ZeroLatencyVoiceChat()
    else:
        chat = SimpleInterruptibleChat()
    
    chat.run()