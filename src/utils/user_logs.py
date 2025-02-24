"""User interaction logging utilities.

This module handles logging of user queries and responses with timestamps.
"""

from datetime import datetime
from pathlib import Path
import json
import os

class UserLogger:
    """Handles logging of user interactions."""

    def __init__(self, username: str):
        """Initialize the logger for a specific user.
        
        Args:
            username: The username to log interactions for
        """
        self.username = username
        self.log_dir = Path("user_logs")
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"{username}_interactions.txt"

    def log_interaction(self, query: str, response: dict, database_name: str = "Unknown") -> None:
        """Log a user interaction with timestamp.
        
        Args:
            query: The user's query
            response: The system's response dictionary
            database_name: Name of the database being queried
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the log entry
        log_entry = {
            "timestamp": timestamp,
            "database": database_name,
            "query": query,
            "answer": response.get("answer", ""),
            "extracted_content": response.get("extracted_content", []),
            "source_count": response.get("source_count", 0)
        }
        
        # Write to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Database: {database_name}\n")
            f.write(f"Query: {query}\n")
            f.write(f"Answer: {response.get('answer', '')}\n")
            f.write("\nExtracted Content:\n")
            for content in response.get("extracted_content", []):
                f.write(f"- {content.get('content', '')}\n")
            f.write(f"\nSource Count: {response.get('source_count', 0)}\n")
            f.write(f"{'='*80}\n")

    def get_user_history(self) -> str:
        """Get the user's interaction history.
        
        Returns:
            The contents of the user's log file
        """
        if self.log_file.exists():
            with open(self.log_file, "r", encoding="utf-8") as f:
                return f.read()
        return "No interaction history found." 