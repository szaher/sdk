from datetime import datetime
from typing import Dict, List, Optional

from podman import PodmanClient

from ..constants import constants
from ..types import types
from .base import TrainingBackend


class PodmanBackend(TrainingBackend):
    """Backend using the Podman SDK to run training containers."""

    def __init__(self, image: str, run_kwargs: Optional[Dict] = None):
        super().__init__(image=image)
        self.client = PodmanClient()
        self.run_kwargs = run_kwargs or {}

    def run(
        self,
        command: List[str],
        args: List[str],
        job_name: str,
        runtime_name: str,
    ) -> int:
        labels = {
            constants.LOCAL_TRAINJOB_LABEL: job_name,
            constants.LOCAL_RUNTIME_LABEL: runtime_name,
        }
        container = self.client.containers.create(
            self.image,
            command + args,
            labels=labels,
            **self.run_kwargs,
        )
        container.start()
        result = container.wait()
        return result.get("StatusCode", 1)

    def _to_trainjob(self, container) -> types.TrainJob:
        created = datetime.fromisoformat(
            container.attrs["Created"].replace("Z", "+00:00")
        )
        runtime_name = container.labels.get(constants.LOCAL_RUNTIME_LABEL, "local")
        runtime = types.TrainingRuntime(
            name=runtime_name, trainer=types.DEFAULT_TRAINER
        )
        status = container.status.capitalize()
        step = types.Step(name=constants.NODE, status=status, pod_name=container.name)
        return types.TrainJob(
            name=container.labels.get(constants.LOCAL_TRAINJOB_LABEL, container.name),
            creation_timestamp=created,
            runtime=runtime,
            steps=[step],
            status=status,
        )

    def list_jobs(self) -> List[types.TrainJob]:
        containers = self.client.containers.list(
            all=True, filters={"label": constants.LOCAL_TRAINJOB_LABEL}
        )
        return [self._to_trainjob(c) for c in containers]

    def get_job(self, name: str) -> types.TrainJob:
        containers = self.client.containers.list(
            all=True,
            filters={"label": f"{constants.LOCAL_TRAINJOB_LABEL}={name}"},
        )
        if not containers:
            raise ValueError(f"Job {name} not found")
        return self._to_trainjob(containers[0])

    def get_job_logs(self, name: str) -> Dict[str, str]:
        containers = self.client.containers.list(
            all=True,
            filters={"label": f"{constants.LOCAL_TRAINJOB_LABEL}={name}"},
        )
        if not containers:
            return {}
        logs = containers[0].logs().decode()
        return {f"{constants.NODE}-0": logs}
