import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MEMORY_FILE = os.path.join(SCRIPT_DIR, "user_memory.json")


class MemoryManager:
    def __init__(self):
        self.short_term_memory = []  
        self.long_term_memory = self._load_long_term_memory()

    def _load_long_term_memory(self) -> dict:
        """Loads user facts from a JSON file."""
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"facts": []}

    def save_long_term_memory(self):
        """Saves updated user facts to the JSON file."""
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.long_term_memory, f, indent=2, ensure_ascii=False)

    def add_fact(self, fact: str):
        """Adds a new fact to long-term memory and saves it."""
        if fact not in self.long_term_memory["facts"]:
            self.long_term_memory["facts"].append(fact)
            self.save_long_term_memory()

    def get_long_term_context(self) -> str:
        """Returns a formatted string of all known user facts."""
        if not self.long_term_memory["facts"]:
            return "No specific user information stored yet."
        return "Known user facts: " + " | ".join(self.long_term_memory["facts"])

    def add_to_short_term(self, role: str, text: str):
        """Adds a message to the short-term conversation history."""
        self.short_term_memory.append({"role": role, "parts": text})
        if len(self.short_term_memory) > 10:
            self.short_term_memory = self.short_term_memory[-10:]

    def get_short_term_history(self) -> list:
        """Returns the current short-term conversation history."""
        return self.short_term_memory

    def clear_short_term(self):
        """Clears the short-term memory (useful for starting a new session)."""
        self.short_term_memory = []