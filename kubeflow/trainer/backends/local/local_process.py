# Copyright 2025 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import string
import tempfile
import uuid
import venv
import random
from typing import List, Optional, Set, Dict

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types
from kubeflow.trainer.backends.local.types import LocalProcessBackendConfig
from kubeflow.trainer.backends.local import utils
from kubeflow.trainer.backends import base
from kubeflow.trainer.backends.local import runtimes as local_runtimes
from kubeflow.trainer.backends.local import job


logger = logging.getLogger(__name__)


class LocalProcessBackend(base.TrainingBackend):

    def __init__(self,
                 cfg: LocalProcessBackendConfig,
                 ):
        # list of running subprocesses
        self.__jobs: List[job.LocalJob] = []
        self.cfg = cfg


    def list_runtimes(self) -> List[types.LocalRuntime]:

        return local_runtimes.runtimes

    def get_runtime(self, name: str) -> Optional[types.LocalRuntime]:

        _runtime = [rt for rt in local_runtimes.runtimes if rt.name == name]
        if not _runtime:
            raise ValueError(f"Runtime '{name}' not found.")

        return _runtime[0]


    def train(self,
              runtime: types.LocalRuntime,
              initializer: Optional[types.Initializer] = None,
              trainer: Optional[types.RuntimeTrainer] = None) -> str:

        train_job_name = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11]
        # Build the env
        if not trainer:
            raise ValueError("Cannot create TrainJob without a Trainer")
        if isinstance(trainer, types.CustomTrainer):
            trainer: types.CustomTrainer = trainer

        # create temp dir for venv
        target_dir = tempfile.mkdtemp(prefix=f"{train_job_name}-")
        runtime.execution_dir = target_dir

        # create venv
        if runtime.create_venv:
            self.__create_venv(env_dir=target_dir)
            runtime.python_path = utils.get_venv_python_path(target_dir)

        command, args = utils.build_local_training_executable(
            runtime,
            trainer.func,
            trainer.func_args,
            trainer.pip_index_url,
            trainer.packages_to_install,
        )

        memory_limit = None
        cpu_limit = None
        cpu_time = None
        nice = 0

        if hasattr(trainer, "resources_per_node"):
            memory_limit = trainer.resources_per_node.get("memory")
            cpu_limit = trainer.resources_per_node.get("cpu")
            cpu_time = trainer.resources_per_node.get("cpu_time")
            nice = trainer.resources_per_node.get("nice")

        # prepare local job
        j = job.LocalJob(
            name=train_job_name,
            command=args,
            cpu_time=cpu_time,
            cpu_limit=cpu_limit,
            mem_limit=memory_limit,
            nice=nice,
        )

        # register job in local list
        self.__jobs.append(j)

        # start job
        j.start()

        return train_job_name

    def __create_venv(self, env_dir: str) -> None:
        """Create Virtual Environment for the Training Job.
        """
        # @szaher do we need to replace this with another LocalJob for Env preparation?
        venv.create(env_dir=env_dir, with_pip=True)

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> List[types.LocalTrainJob]:

        result = [
            types.LocalTrainJob(
                name=j.name, creation_timestamp=j.creation_time,
                runtime=runtime, steps=[], job=j, num_nodes=len(self.__jobs),
            )
            for j in self.__jobs
        ]

        return result

    def get_job(self, name: str) -> Optional[types.LocalTrainJob]:
        j = [j for j in self.__jobs if j.name == name]
        if not j:
            raise ValueError("No TrainJob with name '%s'" % name)
        return types.LocalTrainJob(
            name=j[0].name,
            creation_timestamp=j[0].completion_time,
            runtime=None, steps=[], job=j[0], num_nodes=len(self.__jobs),
        )

    def get_job_logs(self,
                     name: str,
                     follow: Optional[bool] = False,
                     step: str = constants.NODE,
                     node_rank: int = 0) -> Dict[str, str]:
        j = self.get_job(name=name)
        return {
            j.name: j.job.logs(follow=follow),
        }

    def delete_job(self, name: str):
        # delete job from registry or job list
        target = [j for j in self.__jobs if j.name == name]
        if not target:

            raise ValueError("No TrainJob with name '%s'" % name)
        # request process cancellation
        target[0].cancel()
        # remove the job from the list of jobs
        self.__jobs.remove(target[0])

    def wait_for_job_status(
        self,
        name: str,
        status: Set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.LocalTrainJob:

        local_job = self.get_job(name=name)
        local_job.job.join(timeout=timeout)
        return local_job





