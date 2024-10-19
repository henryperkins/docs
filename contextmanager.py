# context_manager.py

import threading
from collections import deque
from typing import List, Dict, Optional

class ContextManager:
    """
    Manages persistent context information for AI interactions.
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.context_entries = deque(maxlen=self.max_entries)
        self.lock = threading.Lock()

    def add_context(self, context_entry: str):
        """
        Adds a new context entry to the context manager.

        Args:
            context_entry (str): The context information to add.
        """
        with self.lock:
            self.context_entries.append(context_entry)

    def get_context(self) -> List[str]:
        """
        Retrieves all current context entries.

        Returns:
            List[str]: A list of context entries.
        """
        with self.lock:
            return list(self.context_entries)

    def clear_context(self):
        """
        Clears all context entries.
        """
        with self.lock:
            self.context_entries.clear()

    def remove_context(self, context_reference: str):
        """
        Removes context entries that contain the specified reference.

        Args:
            context_reference (str): Reference string to identify context entries to remove.
        """
        with self.lock:
            self.context_entries = deque(
                [entry for entry in self.context_entries if context_reference not in entry],
                maxlen=self.max_entries
            )

    def get_relevant_context(self, query: str, top_k: int = 5) -> List[str]:
        """
        Retrieves the most relevant context entries based on the query.

        Args:
            query (str): The query to match against context entries.
            top_k (int): Number of top relevant entries to return.

        Returns:
            List[str]: A list of relevant context entries.
        """
        with self.lock:
            matched_entries = [entry for entry in self.context_entries if query.lower() in entry.lower()]
            return matched_entries[:top_k]