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

"""Utility functions for LiveTrainer CRD generation."""

import inspect
import os
import textwrap

from kubeflow_trainer_api import models

from kubeflow.trainer.backends.kubernetes import utils
from kubeflow.trainer.constants import constants
from kubeflow.trainer.livetrainer.runtime import get_live_trainer_instrumentation_wrapper
from kubeflow.trainer.livetrainer.types import LiveTrainer
from kubeflow.trainer.types import types


def get_trainer_cr_from_live_trainer(
    runtime: types.Runtime,
    trainer: LiveTrainer,
) -> models.TrainerV1alpha1Trainer:
    """Build Trainer CRD from a LiveTrainer configuration.

    Extracts the user's training function, wraps it with hot-reload instrumentation
    code, and builds the trainer command.

    Args:
        runtime: Training runtime configuration.
        trainer: LiveTrainer instance with configuration.

    Returns:
        Trainer CRD with wrapped training function.
    """
    # Ensure runtime trainer has a command set
    try:
        _ = runtime.trainer.command
    except AttributeError:
        if runtime.trainer.framework == "pytorch":
            runtime.trainer.set_command(constants.TORCH_COMMAND)
        else:
            runtime.trainer.set_command(constants.DEFAULT_COMMAND)

    trainer_cr = models.TrainerV1alpha1Trainer()

    # Add number of nodes
    if trainer.num_nodes:
        trainer_cr.num_nodes = trainer.num_nodes

    # Add resources per node
    if trainer.resources_per_node:
        trainer_cr.resources_per_node = utils.get_resources_per_node(trainer.resources_per_node)

    # Add environment variables
    if trainer.env:
        trainer_cr.env = [
            models.IoK8sApiCoreV1EnvVar(name=key, value=value) for key, value in trainer.env.items()
        ]

    # Generate function code
    func_code = inspect.getsource(trainer.func)
    func_code = textwrap.dedent(func_code)

    # Generate function call
    if trainer.func_args is None:
        func_call = f"{trainer.func.__name__}()"
    else:
        func_call = f"{trainer.func.__name__}(**{trainer.func_args})"

    func_code = f"{func_code}\n{func_call}\n"

    # Wrap with LiveTrainer instrumentation
    wrapper_code = get_live_trainer_instrumentation_wrapper(trainer)
    func_code = wrapper_code.replace("{{user_func_import_and_call}}", func_code)

    # Build the command
    func_file = os.path.basename(inspect.getfile(trainer.func))

    # Install Python packages if required
    install_packages = ""
    if trainer.packages_to_install:
        install_packages = utils.get_script_for_python_packages(
            trainer.packages_to_install,
            trainer.pip_index_urls,
        )

    # Build the trainer command with wrapped function code
    command = []
    for c in runtime.trainer.command:
        if "{func_file}" in c:
            exec_script = c.format(func_code=func_code, func_file=func_file)
            if install_packages:
                exec_script = install_packages + exec_script
            command.append(exec_script)
        else:
            command.append(c)

    trainer_cr.command = command

    # Set the TrainJob trainer image if provided
    if trainer.image:
        trainer_cr.image = trainer.image

    return trainer_cr
