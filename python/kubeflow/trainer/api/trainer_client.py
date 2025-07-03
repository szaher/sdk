# Copyright 2024-2025 The Kubeflow Authors.
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
import random
import string
import uuid
from typing import Optional, Union

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types
from kubeflow.trainer.types.backends import BackendConfig
from kubeflow.trainer.backends import TRAINER_BACKEND_REGISTRY


logger = logging.getLogger(__name__)


class TrainerClient:

    def __init__(self, backend_type: Optional[str] = "kubernetes", backend_config: Optional[BackendConfig] = None):
        """
        Initialize a trainer client.

        Args:
            backend_type: name of the backend to be used. default is kubernetes.
            backend_config: backend configuration. default is None.
        """
        backend = self.__init_backend(backend_type, backend_config)
        self.__backend = backend

    def __init_backend(self, backendtype: str, backendconfig: BackendConfig):
        backend = TRAINER_BACKEND_REGISTRY.get(backendtype.lower())
        if not backend:
            raise ValueError("Unknown backend type '{}'".format(backendtype))
        # load the backend class
        backend_cls = backend.get("backend_cls")
        # check if backend configuration is present
        if not backendconfig:
            backendconfig = backend.get("config_cls")()
        # initialize the backend class with the user provided config
        return backend_cls(cfg=backendconfig)

    def list_runtimes(self):
        """List of the available Runtimes.

            Returns:
                List[Runtime]: List of available training runtimes.
                    If no runtimes exist, an empty list is returned.

            Raises:
                TimeoutError: Timeout to list Runtimes.
                RuntimeError: Failed to list Runtimes.
        """
        return self.__backend.list_runtimes()

    def get_runtime(self, name: str):
        return self.__backend.get_runtime(name=name)

    def list_jobs(self, runtime: Optional[types.Runtime] = None):
        """List of all TrainJobs.

            Returns:
                List[TrainerV1alpha1TrainJob]: List of created TrainJobs.
                    If no TrainJob exist, an empty list is returned.

            Raises:
                TimeoutError: Timeout to list TrainJobs.
                RuntimeError: Failed to list TrainJobs.
        """
        return self.__backend.list_jobs(runtime=runtime)

    def get_job(self, name: str):
        """Get the TrainJob object"""
        return self.__backend.get_job(name=name)

    def delete_job(self, name: str):
        """Delete the TrainJob.

            Args:
                name: Name of the TrainJob.

            Raises:
                TimeoutError: Timeout to delete TrainJob.
                RuntimeError: Failed to delete TrainJob.
        """
        return self.__backend.delete_job(name=name)

    def get_job_logs(self,
                     name: str,
                     follow: Optional[bool] = False,
                     step: str = constants.NODE,
                     node_rank: int = 0,
        ):
        """Get the logs from TrainJob"""
        return self.__backend.get_job_logs(name=name, follow=follow, step=step, node_rank=node_rank)

    def train(self,
              runtime: types.Runtime = types.DEFAULT_RUNTIME,
              initializer: Optional[types.Initializer] = None,
              trainer: Optional[Union[types.CustomTrainer, types.BuiltinTrainer]] = None,
        ):
        """
        Create the TrainJob. You can configure these types of training task:
        - Custom Training Task: Training with a self-contained function that encapsulates
            the entire model training process, e.g. `CustomTrainer`.
        - Config-driven Task with Existing Trainer: Training with a trainer that already includes
            the post-training logic, requiring only parameter adjustments, e.g. `BuiltinTrainer`.
        Args:
            runtime (`types.Runtime`): Reference to one of existing Runtimes.
            initializer (`Optional[types.Initializer]`):
                Configuration for the dataset and model initializers.
            trainer (`Optional[types.CustomTrainer, types.BuiltinTrainer]`):
                Configuration for Custom Training Task or Config-driven Task with Builtin Trainer.
        Returns:
            str: The unique name of the TrainJob that has been generated.
        Raises:
            ValueError: Input arguments are invalid.
            TimeoutError: Timeout to create TrainJobs.
            RuntimeError: Failed to create TrainJobs.
        """
        # Generate unique name for the TrainJob.
        # TODO (andreyvelich): Discuss this TrainJob name generation.
        train_job_name = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11]

        self.__backend.train(train_job_name=train_job_name, runtime=runtime, initializer=initializer, trainer=trainer)
