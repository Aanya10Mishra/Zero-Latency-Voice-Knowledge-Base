import re
from typing import List
from groq import AsyncGroq

class VoiceOptimizer:
    """
    Post-processes LLM output for natural speech synthesis.
    - Converts text-heavy responses to spoken English
    - Simplifies complex sentences
    - Adds natural pauses and emphasis markers
    """
    
    def __init__(self, api_key: str):
        self.client = AsyncGroq(api_key=api_key)
        
        # SSML-like markers for TTS
        self.pause_short = "..."
        self.pause_long = ". "
    
    async def optimize_for_voice(self, text: str) -> str:
        """
        Transform LLM response for natural speech.
        """
        # Step 1: Quick rule-based transformations
        text = self._apply_quick_rules(text)
        
        # Step 2: LLM-based simplification for complex text
        if self._is_complex(text):
            text = await self._simplify_with_llm(text)
        
        # Step 3: Add prosody markers
        text = self._add_prosody_markers(text)
        
        return text
    
    def _apply_quick_rules(self, text: str) -> str:
        """Fast rule-based transformations."""
        
        # Remove markdown formatting
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
        text = re.sub(r'`([^`]+)`', r'\1', text)        # Code
        text = re.sub(r'#+\s*', '', text)               # Headers
        
        # Convert bullet points to spoken form
        text = re.sub(r'^\s*[-•]\s*', 'First, ', text, count=1)
        text = re.sub(r'\n\s*[-•]\s*', '. Next, ', text)
        
        # Expand common abbreviations
        abbreviations = {
            r'\be\.g\.\s*': 'for example, ',
            r'\bi\.e\.\s*': 'that is, ',
            r'\betc\.': 'and so on',
            r'\bvs\.': 'versus',
            r'\bapprox\.': 'approximately',
            r'\bmax\.': 'maximum',
            r'\bmin\.': 'minimum',
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Convert numbers to spoken form for better TTS
        text = self._convert_numbers(text)
        
        # Simplify parenthetical content
        text = re.sub(r'\(([^)]{1,30})\)', r', \1,', text)
        text = re.sub(r'\([^)]{30,}\)', '', text)  # Remove long parentheticals
        
        return text
    
    def _convert_numbers(self, text: str) -> str:
        """Convert numbers to spoken-friendly format."""
        # Temperatures
        text = re.sub(r'(\d+)°C', r'\1 degrees Celsius', text)
        text = re.sub(r'(\d+)°F', r'\1 degrees Fahrenheit', text)
        
        # Measurements
        text = re.sub(r'(\d+)\s*mm\b', r'\1 millimeters', text)
        text = re.sub(r'(\d+)\s*cm\b', r'\1 centimeters', text)
        text = re.sub(r'(\d+)\s*m\b', r'\1 meters', text)
        text = re.sub(r'(\d+)\s*kg\b', r'\1 kilograms', text)
        
        # Large numbers
        text = re.sub(r'\b(\d{1,3}),(\d{3}),(\d{3})\b', 
                     lambda m: f"{m.group(1)} million {m.group(2)} thousand {m.group(3)}", text)
        text = re.sub(r'\b(\d{1,3}),(\d{3})\b',
                     lambda m: f"{m.group(1)} thousand {m.group(2)}", text)
        
        return text
    
    def _is_complex(self, text: str) -> bool:
        """Check if text needs LLM simplification."""
        words = text.split()
        if len(words) < 20:
            return False
        
        # Check for long sentences
        sentences = re.split(r'[.!?]', text)
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Check for complex vocabulary (basic heuristic)
        long_words = sum(1 for w in words if len(w) > 10)
        long_word_ratio = long_words / len(words)
        
        return avg_sentence_length > 20 or long_word_ratio > 0.15
    
    async def _simplify_with_llm(self, text: str) -> str:
        """Use LLM to simplify complex text for speech."""
        prompt = f"""Convert this text to natural spoken English. Make it suitable for a voice assistant to read aloud.

Rules:
1. Keep sentences under 15 words
2. Use simple, everyday words
3. Break down complex ideas into smaller parts
4. Remove jargon or explain it simply
5. Use contractions naturally (it's, you'll, don't)
6. Add transition words (so, then, also, now)

Original text:
{text}

Spoken version (only output the converted text):"""

        response = await self.client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast model
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
    
    def _add_prosody_markers(self, text: str) -> str:
        """Add markers for natural speech rhythm."""
        # Add pauses after key phrases
        pause_triggers = [
            r'(First,)',
            r'(Next,)',
            r'(Then,)',
            r'(Finally,)',
            r'(However,)',
            r'(Therefore,)',
            r'(In summary,)',
        ]
        
        for trigger in pause_triggers:
            text = re.sub(trigger, r'\1 ', text)
        
        # Ensure proper spacing around punctuation
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        
        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()