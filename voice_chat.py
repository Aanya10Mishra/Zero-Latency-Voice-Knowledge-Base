import pyttsx3
import requests
import speech_recognition as sr

class VoiceChat:
    def __init__(self):
        # Initialize TTS
        self.tts = pyttsx3.init()
        self.tts.setProperty('rate', 160)
        
        # Initialize speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        self.conversation_id = "voice_session_1"
        self.api_url = "http://localhost:8000/query"
    
    def speak(self, text):
        """Convert text to speech"""
        print(f"🔊 Assistant: {text}")
        self.tts.say(text)
        self.tts.runAndWait()
    
    def stop_speaking(self):
        """Stop any ongoing speech"""
        self.tts.stop()
    
    def listen(self):
        """Listen for voice input"""
        print("🎤 Listening... (speak now)")
        
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"❌ Speech recognition error: {e}")
            return None
    
    def is_quit_command(self, text):
        """Check if user wants to quit"""
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
            return f"Sorry, I couldn't connect to the server: {e}"
    
    def run(self):
        """Main chat loop"""
        self.speak("Hello! I'm your technical assistant. Ask me anything about the manual. Say 'quit' to exit.")
        
        while True:
            # Listen for input
            user_input = self.listen()
            
            if user_input is None:
                self.speak("I didn't catch that. Please try again.")
                continue
            
            # CHECK FOR QUIT FIRST
            if self.is_quit_command(user_input):
                self.stop_speaking()  # Stop any ongoing speech
                self.speak("Goodbye! Have a great day.")
                print("\n👋 Session ended.")
                break
            
            # Process the question
            self.speak("Let me look that up for you...")
            answer = self.query_rag(user_input)
            self.speak(answer)


if __name__ == "__main__":
    print("="*50)
    print("VOICE CHAT - Technical Manual Assistant")
    print("="*50)
    print("\nMake sure the server is running (python main.py)")
    print("Say 'quit' to exit\n")
    
    chat = VoiceChat()
    chat.run()