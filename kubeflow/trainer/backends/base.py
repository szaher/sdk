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

import abc

from typing import Dict, List, Optional, Set
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


class TrainingBackend(abc.ABC):

    @abc.abstractmethod
    def list_runtimes(self) -> types.RuntimeList:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_runtime(self, name: str) -> Optional[types.TrainingRuntime]:
        raise NotImplementedError()

    @abc.abstractmethod
    def train(self,
              runtime: types.Runtime,
              initializer: Optional[types.Initializer] = None,
              trainer: Optional[types.RuntimeTrainer] = None,
              ) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def list_jobs(
            self, runtime: Optional[types.Runtime] = None
    ) -> List[types.TrainJobLike]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job(self, name: str) -> Optional[types.TrainJobLike]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_job_logs(self,
                     name: str,
                     follow: Optional[bool] = False,
                     step: str = constants.NODE,
                     node_rank: int = 0,
                     ) -> Dict[str, str]:
        raise NotImplementedError()

    @abc.abstractmethod
    def delete_job(self, name: str) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def wait_for_job_status(
            self,
            name: str,
            status: Set[str] = {constants.TRAINJOB_COMPLETE},
            timeout: int = 600,
            polling_interval: int = 2,
    ) -> types.TrainJobLike:
        raise NotImplementedError()

    def get_runtime_packages(self, runtime: types.TrainingRuntime):
        raise NotImplementedError()
