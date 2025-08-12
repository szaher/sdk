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

import inspect
import logging
import os
import sys
from pathlib import Path
import textwrap
from typing import List, Callable, Optional, Dict, Any, Tuple
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import local as local_types

logger = logging.getLogger(__name__)


# @szaher we don't need this one but could be useful for discussion
def setup_kubeflow_local_config_dir():
    logger.debug("Setting up kubeflow local config dir %s", constants.DEFAULT_CFG_DIR)

    def _create_dir(dir_name: str, base_dir: str = ""):
        if base_dir:
            dir_name = os.path.join(base_dir, dir_name)
        os.makedirs(dir_name, exist_ok=True)

    cfg_dirs = constants.DEFAULT_CFG_SUB_DIRS.copy()

    # create base directory
    _create_dir(constants.DEFAULT_CFG_DIR)

    # create nested directories
    [
        _create_dir(dir_name=_dir, base_dir=constants.DEFAULT_CFG_DIR) for _dir in cfg_dirs
    ]


def get_venv_python_path(venv_dir: str) -> str:
    venv_path = Path(venv_dir)
    if os.name == 'nt':
        # Windows
        python_exe = venv_path / "Scripts" / "python.exe"
    else:
        # Unix / macOS
        python_exe = venv_path / "bin" / "python"

    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found in virtualenv at: {python_exe}")

    return str(python_exe)


def get_script_for_local_python_packages_install(
    packages_to_install: List[str],
    pip_index_url: str,
    as_user: str = None,
    python_binary: str = sys.executable,
) -> str:
    """
    Get init script to install Python packages from the given pip index URL.
    """
    packages_str = " ".join([str(package) for package in packages_to_install])

    script_for_python_packages = textwrap.dedent(
        """
        if ! [ -x "$(command -v pip)" ]; then
            {python_binary} -m ensurepip || {python_binary} -m ensurepip --user
        fi

        PIP_DISABLE_PIP_VERSION_CHECK=1 {python_binary} -m pip install  --quiet \
        --no-warn-script-location --index-url {index_url} {packages} {as_user}
        """.format(
            python_binary=python_binary,
            index_url=pip_index_url,
            packages=packages_str,
            as_user="--user" if as_user else "",
        )
    )

    return script_for_python_packages




def build_local_training_executable(
    runtime: local_types.LocalRuntime,
    train_func: Callable,
    train_func_parameters: Optional[Dict[str, Any]],
    pip_index_url: str,
    packages_to_install: Optional[List[str]] = None,
) -> Tuple[str, List[str]]:
    """
    Get the Trainer command and args from the given training function and parameters.
    """
    # Check if training function is callable.
    if not callable(train_func):
        raise ValueError(
            f"Training function must be callable, got function type: {type(train_func)}"
        )

    # Extract the function implementation.
    func_code = inspect.getsource(train_func)

    # Extract the file name where the function is defined.
    func_file = os.path.basename(inspect.getfile(train_func))

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

    command = runtime.get_executable_command()

    exec_script = textwrap.dedent(
        """
                read -r -d '' SCRIPT << EOM\n
                {func_code}
                EOM
                printf "%s" \"$SCRIPT\" > \"{func_file}\"
                {python_entrypoint} \"{func_file}\""""
    )

    # Add function code to the execute script.
    exec_script = exec_script.format(
        func_code=func_code,
        func_file=func_file,
        python_entrypoint=command,
    )

    # Install Python packages if that is required.
    if packages_to_install is not None:
        exec_script = (
            get_script_for_local_python_packages_install(
                packages_to_install=packages_to_install,
                pip_index_url=pip_index_url,
                python_binary=runtime.python_path
            )
            + exec_script
        )

    # Return container command and args to execute training function.
    return command, [exec_script]


