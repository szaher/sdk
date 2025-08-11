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

import os
from typing import List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import sys

from kubeflow.trainer.types import types
from kubeflow.trainer.local.job import LocalJob


@dataclass
class LocalRuntime(types.Runtime):
    create_venv: Optional[bool] = True
    command: List[str] = field(default_factory=list)
    python_path: Optional[str] = sys.executable
    execution_dir: Optional[str] = None

    def get_executable_command(self) -> str:
        venv_path = Path(self.execution_dir)
        command_str = " ".join(self.command).lstrip()
        if self.create_venv:
            if os.name == 'nt':
                # Windows
                command_exe = venv_path / "Scripts" / command_str
            else:
                # Unix / macOS
                command_exe = venv_path / "bin" / command_str
        else:
            command_exe = command_str

        # @szaher need to make sure venv is created before this check
        # if not command_exe.exists():
        #     raise FileNotFoundError(f"Python executable not found in virtualenv at: {command_exe}")

        return str(command_exe)


@dataclass
class LocalTrainJob(types.TrainJob):
    job: LocalJob = None
