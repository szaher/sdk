from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from kubeflow.trainer.types import types


class TrainingBackend(ABC):
    """Abstract base class for trainer execution backends."""

    def __init__(self, image: Optional[str] = None):
        self.image = image

    @abstractmethod
    def run(
        self,
        command: List[str],
        args: List[str],
        job_name: str,
        runtime_name: str,
    ) -> int:
        """Run the provided command.

        Args:
            command: The container command or entrypoint.
            args: Arguments for the command.

        Returns:
            Exit code from the execution process.
        """
        raise NotImplementedError

    def list_jobs(self) -> List["types.TrainJob"]:
        raise NotImplementedError

    def get_job(self, name: str) -> "types.TrainJob":
        raise NotImplementedError

    def get_job_logs(self, name: str) -> Dict[str, str]:
        raise NotImplementedError
