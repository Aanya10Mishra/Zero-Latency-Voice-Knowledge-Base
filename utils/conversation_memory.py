from collections import defaultdict
from typing import List, Dict
from dataclasses import dataclass, field
import time

@dataclass
class ConversationTurn:
    user: str
    assistant: str
    timestamp: float = field(default_factory=time.time)
    entities: List[str] = field(default_factory=list)

class ConversationMemory:
    """
    Manages conversation history for context resolution.
    Tracks entities mentioned for quick reference resolution.
    """
    
    def __init__(self, max_turns: int = 10):
        self.conversations: Dict[str, List[ConversationTurn]] = defaultdict(list)
        self.entity_index: Dict[str, List[str]] = defaultdict(list)  # entity -> [conv_id]
        self.max_turns = max_turns
    
    def add_turn(self, conversation_id: str, user: str, assistant: str):
        """Add a conversation turn and extract entities."""
        entities = self._extract_entities(user + " " + assistant)
        
        turn = ConversationTurn(
            user=user,
            assistant=assistant,
            entities=entities
        )
        
        self.conversations[conversation_id].append(turn)
        
        # Update entity index for quick lookups
        for entity in entities:
            self.entity_index[entity].append(conversation_id)
        
        # Trim old turns
        if len(self.conversations[conversation_id]) > self.max_turns:
            self.conversations[conversation_id] = \
                self.conversations[conversation_id][-self.max_turns:]
    
    def get_history(self, conversation_id: str) -> List[dict]:
        """Get conversation history as list of dicts."""
        return [
            {"user": t.user, "assistant": t.assistant}
            for t in self.conversations[conversation_id]
        ]
    
    def get_recent_entities(self, conversation_id: str, n: int = 5) -> List[str]:
        """Get recently mentioned entities for quick resolution."""
        entities = []
        for turn in reversed(self.conversations[conversation_id][-n:]):
            entities.extend(turn.entities)
        return list(dict.fromkeys(entities))  # Preserve order, remove duplicates
    
    def _extract_entities(self, text: str) -> List[str]:
        """Simple entity extraction (can be enhanced with NER)."""
        # For technical manuals: extract part numbers, component names, etc.
        import re
        
        entities = []
        
        # Part numbers (e.g., "XR-500", "Model A123")
        part_numbers = re.findall(r'\b[A-Z]{1,3}[-]?\d{2,5}[A-Z]?\b', text)
        entities.extend(part_numbers)
        
        # Component names (capitalized multi-word terms)
        components = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
        entities.extend(components)
        
        return entities