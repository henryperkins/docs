"""
context_manager.py

This module defines the ContextManager class, which manages persistent context information for AI interactions. It uses a deque to store context entries and a threading lock to ensure thread-safe operations.
"""

import threading
from collections import deque
from typing import List

class ContextManager:
    """
    Manages persistent context information for AI interactions.
    """

    def __init__(self, max_entries: int = 100):
        """
        Initializes the ContextManager with a maximum number of entries it can hold.

        Args:
            max_entries (int): The maximum number of context entries allowed.
        """
        self.max_entries = max_entries
        self.context_entries = deque(maxlen=self.max_entries)  # Use deque for efficient appends and pops
        self.lock = threading.Lock()  # Ensure thread-safe operations

    def add_context(self, context_entry: str):
        """
        Adds a new context entry to the context manager.

        Args:
            context_entry (str): The context information to add.
        """
        with self.lock:  # Lock to ensure thread-safe access
            self.context_entries.append(context_entry)

    def get_context(self) -> List[str]:
        """
        Retrieves all current context entries.

        Returns:
            List[str]: A list of context entries.
        """
        with self.lock:  # Lock to ensure thread-safe access
            return list(self.context_entries)

    def clear_context(self):
        """
        Clears all context entries.
        """
        with self.lock:  # Lock to ensure thread-safe access
            self.context_entries.clear()

    def remove_context(self, context_reference: str):
        """
        Removes context entries that contain the specified reference.

        Args:
            context_reference (str): Reference string to identify context entries to remove.
        """
        with self.lock:  # Lock to ensure thread-safe access
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
        with self.lock:  # Lock to ensure thread-safe access
            matched_entries = [entry for entry in self.context_entries if query.lower() in entry.lower()]
            return matched_entries[:top_k]