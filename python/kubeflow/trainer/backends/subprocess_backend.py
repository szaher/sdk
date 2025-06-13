import subprocess
from typing import List, Optional

from .base import TrainingBackend


class SubprocessBackend(TrainingBackend):
    """Backend that runs commands directly using subprocess."""

    def __init__(self, env: Optional[dict] = None):
        super().__init__(image=None)
        self.env = env

    def run(self, command: List[str], args: List[str]) -> int:
        proc = subprocess.run(command + args, env=self.env, check=False)
        return proc.returncode
