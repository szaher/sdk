import subprocess
from datetime import datetime
from typing import Dict, List, Optional

from ..constants import constants
from ..types import types
from .base import TrainingBackend


class SubprocessBackend(TrainingBackend):
    """Backend that runs commands directly using subprocess."""

    def __init__(self, env: Optional[dict] = None):
        super().__init__(image=None)
        self.env = env
        self._jobs: Dict[str, subprocess.CompletedProcess] = {}
        self._start_times: Dict[str, datetime] = {}
        self._runtime_names: Dict[str, str] = {}

    def run(
        self,
        command: List[str],
        args: List[str],
        job_name: str,
        runtime_name: str,
    ) -> int:
        proc = subprocess.run(
            command + args,
            env=self.env,
            check=False,
            capture_output=True,
            text=True,
        )
        self._jobs[job_name] = proc
        self._start_times[job_name] = datetime.utcnow()
        self._runtime_names[job_name] = runtime_name
        return proc.returncode

    def _to_trainjob(
        self, name: str, proc: subprocess.CompletedProcess
    ) -> types.TrainJob:
        created = self._start_times.get(name, datetime.utcnow())
        runtime_name = self._runtime_names.get(name, "local")
        runtime = types.TrainingRuntime(
            name=runtime_name, trainer=types.DEFAULT_TRAINER
        )
        status = "Succeeded" if proc.returncode == 0 else "Failed"
        step = types.Step(name=constants.NODE, status=status, pod_name=name)
        return types.TrainJob(
            name=name,
            creation_timestamp=created,
            runtime=runtime,
            steps=[step],
            status=status,
        )

    def list_jobs(self) -> List[types.TrainJob]:
        return [self._to_trainjob(n, p) for n, p in self._jobs.items()]

    def get_job(self, name: str) -> types.TrainJob:
        if name not in self._jobs:
            raise ValueError(f"Job {name} not found")
        return self._to_trainjob(name, self._jobs[name])

    def get_job_logs(self, name: str) -> Dict[str, str]:
        if name not in self._jobs:
            return {}
        proc = self._jobs[name]
        logs = (proc.stdout or "") + (proc.stderr or "")
        return {f"{constants.NODE}-0": logs}
