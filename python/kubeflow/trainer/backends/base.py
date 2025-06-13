from abc import ABC, abstractmethod
from typing import List, Optional


class TrainingBackend(ABC):
    """Abstract base class for trainer execution backends."""

    def __init__(self, image: Optional[str] = None):
        self.image = image

    @abstractmethod
    def run(self, command: List[str], args: List[str]) -> int:
        """Run the provided command.

        Args:
            command: The container command or entrypoint.
            args: Arguments for the command.

        Returns:
            Exit code from the execution process.
        """
        raise NotImplementedError
