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

    def get_runtime(self, name: str) -> Optional[local_types.LocalRuntime]:
        """Get the the Runtime object
            Returns:
                LocalRuntime: Runtime object for the given name.
            Raises:
                ValueError: If no Runtime is found for the given name.
        """
        _runtime = [rt for rt in local_runtimes.runtimes if rt.name == name]
        if not _runtime:
            raise ValueError(f"Runtime '{name}' not found.")

        return _runtime[0]


    def train(self, train_job_name:str,
              runtime: local_types.LocalRuntime,
              initializer: Optional[types.Initializer] = None,
              trainer: Optional[types.Trainer] = None) -> str:
        """
                Create the LocalTrainJob. You can configure these types of training task:

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
        """Create Virtual Environment for the Training Job.
        """
        # @szaher do we need to replace this with another LocalJob for Env preparation?
        venv.create(env_dir=env_dir, with_pip=True)

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> List[local_types.LocalTrainJob]:
        """List of all TrainJobs.

                Returns:
                    List[LocalTrainJob]: List of created LocalTrainJobs.
                        If no TrainJob exist, an empty list is returned.
                """

        result = [
            local_types.LocalTrainJob(
                name=j.name, creation_timestamp=j.creation_time,
                runtime=runtime, steps=[], job=j,
            )
            for j in self.__jobs
        ]

        return result

    def get_job(self, name: str) -> Optional[local_types.LocalTrainJob]:
        """Get the TrainJob object.
            Returns:
                LocalTrainJob: LocalTrainJob object.
            Raises:
                ValueError: if TrainJob does not exist.
        """
        j = [j for j in self.__jobs if j.name == name]
        if not j:
            raise ValueError("No TrainJob with name '%s'" % name)
        return local_types.LocalTrainJob(
            name=j[0].name,
            creation_timestamp=j[0].completion_time,
            runtime=None, steps=[], job=j[0]
        )

    def get_job_logs(self,
                     name: str,
                     follow: Optional[bool] = False,
                     step: str = constants.NODE,
                     node_rank: int = 0) -> List[str]:
        """Get the logs from TrainJob
            Args:
                  name (`str`) : The name of the TrainJob.
                  follow (`Optional[bool]`): Follow the log stream or not (default: False).
                  step (`str`): Step number (default: 0) [NOT IMPLEMENTED].
                  node_rank (`int`): Node rank (default: 0) [NOT IMPLEMENTED].
            Returns:
                List[str]: List of logs from TrainJob.
            Raises:
                ValueError: if TrainJob does not exist.
        """
        j = self.get_job(name=name)
        return j.job.logs(follow=follow)

    def delete_job(self, name: str) -> None:
        """Delete the TrainJob.

                Args:
                    name: Name of the TrainJob.

                Raises:
                    ValueError: if TrainJob does not exist.
                """

        # delete job from registry or job list
        target = [j for j in self.__jobs if j.name == name]
        if not target:

            raise ValueError("No TrainJob with name '%s'" % name)
        # request process cancellation
        target[0].cancel()
        # remove the job from the list of jobs
        self.__jobs.remove(target[0])






