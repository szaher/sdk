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

import typing
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel
from kubeflow_trainer_api.models import trainer_v1alpha1_ml_policy

from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.types import types


class LocalProcessBackendConfig(BaseModel):
    cleanup: bool = True
    # change working directory to venv for execution.
    run_in_venv_dir: bool = True


class LocalRuntime(BaseModel):
    runtime: types.Runtime = None
    ml_policy: trainer_v1alpha1_ml_policy.TrainerV1alpha1MLPolicy

    class Config:
        arbitrary_types_allowed = True


class LocalBackendStep(BaseModel):
    step_name: str
    job: LocalJob

    class Config:
        arbitrary_types_allowed = True


class LocalBackendJobs(BaseModel):
    steps: Optional[List[LocalBackendStep]] = []
    runtime: Optional[LocalRuntime] = None
    name: str
    created: typing.Optional[datetime] = None
    completed: typing.Optional[datetime] = None

    class Config:
        arbitrary_types_allowed = True
