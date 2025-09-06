import inspect
import os
import textwrap
from pathlib import Path
from string import Template
from tkinter import Listbox
from typing import List, Callable, Optional, Dict, Any

from kubeflow_trainer_api import models

from kubeflow.trainer.constants import constants
from kubeflow.trainer.constants import local_exec_constants
from kubeflow.trainer.constants.constants import GPU_LABEL
from kubeflow.trainer.types import types


def get_runtime_trainer(
        venv_dir: str,
        python_bin: str,
        framework: str,
        ml_policy: models.TrainerV1alpha1MLPolicy,
) -> types.RuntimeTrainer:
    """
    Get the RuntimeTrainer object.
    """

    trainer = types.RuntimeTrainer(
        trainer_type=(
            types.TrainerType.BUILTIN_TRAINER
            if framework == types.TORCH_TUNE
            else types.TrainerType.CUSTOM_TRAINER
        ),
        framework=framework,
    )

    # set command to run from venv
    venv_bin_dir = str(Path(venv_dir) / "bin")
    default_cmd = [str(Path(venv_bin_dir) / local_exec_constants.DEFAULT_COMMAND)]
    # Set the Trainer entrypoint.
    if ml_policy.torch:
        _c = [os.path.join(venv_bin_dir, local_exec_constants.TORCH_COMMAND)]
        trainer.set_command(tuple(_c))
    elif ml_policy.mpi:
        # mpi isn't supported yet
        trainer.set_command(tuple(default_cmd))
    else:
        trainer.set_command(tuple(default_cmd))

    return trainer


def get_dependencies_command(python_bin, pip_bin: str, pip_index_urls: str, packages: List[str]):
    options = [f"--index-url {pip_index_urls[0]}"]
    options.extend(f"--extra-index-url {extra_index_url}" for extra_index_url in pip_index_urls[1:])

    mapping = {
        "PYTHON_BIN": python_bin,
        "PIP_BIN": pip_bin,
        "PIP_INDEX": " ".join(options),
        "PACKAGE_STR": " ".join(packages),
    }
    t = Template(local_exec_constants.DEPENDENCIES_SCRIPT)
    result = t.substitute(**mapping)
    return (
        'bash',
        '-c',
        result
    )


def get_local_devices(resources: dict[str, str]) -> (str, int):
    device, device_count = constants.UNKNOWN, 0

    if constants.GPU_LABEL in resources.items():
        device = constants.GPU_LABEL.split("/")[1]
        device_count = resources[constants.GPU_LABEL]
    elif constants.TPU_LABEL in resources.items():
        device = constants.TPU_LABEL.split("/")[1]
        device_count = resources[constants.TPU_LABEL]
    elif constants.MPS_LABEL in resources.items():
        device = constants.MPS_LABEL.split("/")[1]
        device_count = resources[constants.MPS_LABEL]
    return device, device_count


def get_command_using_train_func(
        runtime: types.Runtime,
        train_func: Callable,
        train_func_parameters: Optional[Dict[str, Any]],
        venv_dir: str,
        train_job_name: str,
) -> tuple:
    """
    Get the Trainer container command from the given training function and parameters.
    """
    # Check if the runtime has a Trainer.
    if not runtime.trainer:
        raise ValueError(f"Runtime must have a trainer: {runtime}")

    # Check if training function is callable.
    if not callable(train_func):
        raise ValueError(
            f"Training function must be callable, got function type: {type(train_func)}"
        )

    # Extract the function implementation.
    func_code = inspect.getsource(train_func)

    # Extract the file name where the function is defined and move it the venv directory.
    func_file = Path(venv_dir) / "{}-{}".format(train_job_name, os.path.basename(inspect.getfile(train_func)))

    # Function might be defined in some indented scope (e.g. in another function).
    # We need to dedent the function code.
    func_code = textwrap.dedent(func_code)

    # Wrap function code to execute it from the file. For example:
    # TODO (andreyvelich): Find a better way to run users' scripts.
    # def train(parameters):
    #     print('Start Training...')
    # train({'lr': 0.01})
    if train_func_parameters is None:
        func_code = f"{func_code}\n{train_func.__name__}()\n"
    else:
        func_code = f"{func_code}\n{train_func.__name__}({train_func_parameters})\n"

    with open(func_file, "w") as f:
        f.write(func_code)
    f.close()

    t = Template(local_exec_constants.LOCAL_EXEC_JOB_SCRIPT)
    mapping = {
        "PARAMETERS": "",
        "PYENV_LOCATION": venv_dir,
        "ENTRYPOINT": " ".join(runtime.trainer.command),
        "FUNC_FILE": func_file,
    }
    command = t.safe_substitute(**mapping)

    return 'bash', '-c', command


def get_cleanup_command(venv_dir: str) -> tuple:
    mapping = {"PYENV_LOCATION": venv_dir}
    t = Template(local_exec_constants.LOCAL_EXEC_JOB_CLEANUP_SCRIPT)
    cleanup_command = t.substitute(**mapping)

    return 'bash', '-c', cleanup_command