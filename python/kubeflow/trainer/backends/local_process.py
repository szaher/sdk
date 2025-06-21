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
import datetime
import logging
import tempfile
import venv
from typing import List, Optional

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types, local as local_types
from kubeflow.trainer.types.backends import LocalProcessBackendConfig
from kubeflow.trainer.utils import utils, local as local_utils
from kubeflow.trainer.backends import base
from kubeflow.trainer.local import runtimes as local_runtimes
from kubeflow.trainer.local import job


logger = logging.getLogger(__name__)


class LocalProcessBackend(base.TrainingBackend):

    def __init__(self,
                 cfg: LocalProcessBackendConfig,
                 ):
        # list of running subprocesses
        self.__jobs: List[job.LocalJob] = []
        self.cfg = cfg


    def list_runtimes(self) -> List[types.Runtime]:
        """List of the available Runtimes.

               Returns:
                   List[Runtime]: List of available local training runtimes.
                       If no runtimes exist, an empty list is returned.
               """

        return local_runtimes.runtimes

    def get_runtime(self, name: str) -> Optional[types.Runtime]:
        """Get the the Runtime object"""
        if name not in local_runtimes:
            raise ValueError(f"Runtime '{name}' not found.")

        return local_runtimes[name]


    def train(self, train_job_name:str,
              runtime: local_types.LocalRuntime,
              initializer: Optional[types.Initializer] = None,
              trainer: Optional[types.Trainer] = None) -> str:
        """
                Create the TrainJob. You can configure these types of training task:

                - Custom Training Task: Training with a self-contained function that encapsulates
                    the entire model training process, e.g. `CustomTrainer`.

                Args:
                    train_job_name: The name of the training job.
                    runtime (`types.Runtime`): Reference to one of existing Runtimes.
                    initializer (`Optional[types.Initializer]`):
                        Configuration for the dataset and model initializers.
                    trainer (`Optional[types.CustomTrainer]`):
                        Configuration for Custom Training Task.

                Returns:
                    str: The unique name of the TrainJob that has been generated.

                Raises:
                    ValueError: Input arguments are invalid.
                    TimeoutError: Timeout to create TrainJobs.
                    RuntimeError: Failed to create TrainJobs.
                """
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
            runtime.python_path = local_utils.get_venv_python_path(target_dir)

        command, args = local_utils.build_local_training_executable(
            runtime,
            trainer.func,
            trainer.func_args,
            trainer.pip_index_url,
            trainer.packages_to_install,
        )

        # prepare local job
        j = job.LocalJob(
            name=train_job_name,
            command=args,
        )

        # register job in local list
        self.__jobs.append(j)

        # start job
        j.start()

        return train_job_name

    def __create_venv(self, env_dir: str) -> None:
        venv.create(env_dir=env_dir, with_pip=True)

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> List[local_types.LocalTrainJob]:
        """List of all TrainJobs.

                Returns:
                    List[TrainerV1alpha1TrainJob]: List of created TrainJobs.
                        If no TrainJob exist, an empty list is returned.

                Raises:
                    TimeoutError: Timeout to list TrainJobs.
                    RuntimeError: Failed to list TrainJobs.
                """

        result = [local_types.LocalTrainJob(name=j.name, creation_timestamp=datetime.datetime.now(), runtime=runtime, steps=[], job=j) for j in self.__jobs]

        return result

    def get_job(self, name: str) -> Optional[local_types.LocalTrainJob]:
        """Get the TrainJob object"""
        j = [j for j in self.__jobs if j.name == name]
        if not j:
            raise ValueError("No TrainJob with name '%s'" % name)
        return local_types.LocalTrainJob(
            name=j[0].name,
            creation_timestamp=datetime.datetime.now(),
            runtime=None, steps=[], job=j[0]
        )

    def get_job_logs(self,
                     name: str,
                     follow: Optional[bool] = False,
                     step: str = constants.NODE,
                     node_rank: int = 0) -> List[str]:
        """Get the logs from TrainJob"""
        j = self.get_job(name=name)
        lines = []
        for line in j.job.follow_logs():
            print(line)
            lines.append(line)
        return lines



    def delete_job(self, name: str) -> None:
        """Delete the TrainJob.

                Args:
                    name: Name of the TrainJob.

                Raises:
                    TimeoutError: Timeout to delete TrainJob.
                    RuntimeError: Failed to delete TrainJob.
                """

        # delete job from registry or job list
        target = [j for j in self.__jobs if j.name == name]
        if not target:

            raise ValueError("No TrainJob with name '%s'" % name)
        self.__jobs.remove(target[0])






