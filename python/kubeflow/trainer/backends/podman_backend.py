import subprocess
from typing import List, Optional

from .base import TrainingBackend


class PodmanBackend(TrainingBackend):
    """Backend using podman CLI to run training containers."""

    def __init__(self, image: str, additional_args: Optional[List[str]] = None):
        super().__init__(image=image)
        self.additional_args = additional_args or []

    def run(self, command: List[str], args: List[str]) -> int:
        podman_cmd = (
            [
                "podman",
                "run",
                "--rm",
            ]
            + self.additional_args
            + [self.image]
            + command
            + args
        )
        proc = subprocess.run(podman_cmd, check=False)
        return proc.returncode
