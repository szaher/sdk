import inspect
import os
import shutil

import textwrap
from pathlib import Path
from string import Template
from typing import List, Callable, Optional, Dict, Any

from kubeflow_trainer_api import models

from kubeflow.trainer.backends.localprocess import constants as local_exec_constants
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


def get_runtime_trainer(
    venv_dir: str,
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
    else:
        trainer.set_command(tuple(default_cmd))

    return trainer


def get_dependencies_command(pip_index_urls: str, packages: List[str], quiet: bool = True) -> str:
    options = [f"--index-url {pip_index_urls[0]}"]
    options.extend(f"--extra-index-url {extra_index_url}" for extra_index_url in pip_index_urls[1:])

    """
           PIP_DISABLE_PIP_VERSION_CHECK=1 pip install $QUIET $AS_USER \
       --no-warn-script-location $PIP_INDEX $PACKAGE_STR
       """
    mapping = {
        "QUIET": "--quiet" if quiet else "",
        "PIP_INDEX": " ".join(options),
        "PACKAGE_STR": " ".join(packages),
    }
    t = Template(local_exec_constants.DEPENDENCIES_SCRIPT)
    result = t.substitute(**mapping)
    return result


def get_command_using_train_func(
    runtime: types.Runtime,
    train_func: Callable,
    train_func_parameters: Optional[Dict[str, Any]],
    venv_dir: str,
    train_job_name: str,
) -> str:
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
    func_file = Path(venv_dir) / local_exec_constants.LOCAL_EXEC_FILENAME.format(train_job_name)

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

    t = Template(local_exec_constants.LOCAL_EXEC_ENTRYPOINT)
    mapping = {
        "PARAMETERS": "",  ## Torch Parameters if any
        "PYENV_LOCATION": venv_dir,
        "ENTRYPOINT": " ".join(runtime.trainer.command),
        "FUNC_FILE": func_file,
    }
    entrypoint = t.safe_substitute(**mapping)

    return entrypoint


def get_cleanup_script(venv_dir: str, cleanup: bool = True) -> str:
    script = "\n"
    if not cleanup:
        return script

    t = Template(local_exec_constants.LOCAL_EXEC_JOB_CLEANUP_SCRIPT)
    mapping = {
        "PYENV_LOCATION": venv_dir,
    }
    return t.substitute(**mapping)


def get_training_job_command(
    train_job_name: str,
    venv_dir: str,
    trainer: types.CustomTrainer,
    runtime: types.Runtime,
    cleanup: bool = True,
) -> tuple:
    # use local-exec train job template
    t = Template(local_exec_constants.LOCAL_EXEC_JOB_TEMPLATE)
    # find os python binary to create venv
    python_bin = shutil.which("python")
    if not python_bin:
        python_bin = shutil.which("python3")
    if not python_bin:
        raise ValueError("No python executable found")

    # workout if dependencies needs to be installed
    dependency_script = "\n"
    if trainer.packages_to_install:
        dependency_script = get_dependencies_command(
            pip_index_urls=trainer.pip_index_urls
            if trainer.pip_index_urls
            else constants.DEFAULT_PIP_INDEX_URLS,
            packages=trainer.packages_to_install,
            quiet=False,
        )

    entrypoint = get_command_using_train_func(
        venv_dir=venv_dir,
        runtime=runtime,
        train_func=trainer.func,
        train_func_parameters=trainer.func_args,
        train_job_name=train_job_name,
    )

    cleanup_script = get_cleanup_script(cleanup=cleanup, venv_dir=venv_dir)

    mapping = {
        "OS_PYTHON_BIN": python_bin,
        "PYENV_LOCATION": venv_dir,
        "DEPENDENCIES_SCRIPT": dependency_script,
        "ENTRYPOINT": entrypoint,
        "CLEANUP_SCRIPT": cleanup_script,
    }

    command = t.safe_substitute(**mapping)
    return "bash", "-c", command
