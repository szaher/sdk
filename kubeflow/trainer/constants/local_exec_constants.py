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

import textwrap


# The exec script to embed training function into container command.
DEPENDENCIES_SCRIPT = textwrap.dedent(
    """
        $PYTHON_BIN -m ensurepip --upgrade --default-pip
        PIP_DISABLE_PIP_VERSION_CHECK=1 $PIP_BIN install --quiet \
        --no-warn-script-location $PIP_INDEX $PACKAGE_STR
    """
)

# activate virtualenv, then run the entrypoint from the virtualenv bin
LOCAL_EXEC_JOB_SCRIPT = textwrap.dedent(
    """
    source $PYENV_LOCATION/bin/activate
    $ENTRYPOINT "$FUNC_FILE" "$PARAMETERS"
    """
)

TORCH_COMMAND = "torchrun"

# default command, will run from within the virtualenv
DEFAULT_COMMAND = "python"

# remove virtualenv after training is completed.
LOCAL_EXEC_JOB_CLEANUP_SCRIPT = textwrap.dedent(
    """
    rm -rf $PYENV_LOCATION
    """
)