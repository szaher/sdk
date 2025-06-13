from typing import Dict, List, Optional

from podman import PodmanClient

from .base import TrainingBackend


class PodmanBackend(TrainingBackend):
    """Backend using the Podman SDK to run training containers."""

    def __init__(self, image: str, run_kwargs: Optional[Dict] = None):
        super().__init__(image=image)
        self.client = PodmanClient()
        self.run_kwargs = run_kwargs or {}

    def run(self, command: List[str], args: List[str]) -> int:
        container = self.client.containers.create(
            self.image,
            command + args,
            **self.run_kwargs,
        )
        container.start()
        result = container.wait()
        container.remove()
        return result.get("StatusCode", 1)
