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
from kubeflow_trainer_api.models.trainer_v1alpha1_ml_policy import TrainerV1alpha1MLPolicy
from kubeflow_trainer_api.models.trainer_v1alpha1_torch_ml_policy_source import (
    TrainerV1alpha1TorchMLPolicySource,
)
from kubeflow_trainer_api.models.trainer_v1alpha1_torch_elastic_policy import (
    TrainerV1alpha1TorchElasticPolicy,
)
from kubeflow_trainer_api.models.io_k8s_apimachinery_pkg_util_intstr_int_or_string import (
    IoK8sApimachineryPkgUtilIntstrIntOrString,
)
from kubeflow.trainer.types import types as base_types
from kubeflow.trainer.constants import constants
from kubeflow.trainer.backends.localprocess import types


local_runtimes = [
    types.LocalRuntime(
        runtime=base_types.Runtime(
            name=constants.TORCH_RUNTIME,
            trainer=base_types.RuntimeTrainer(
                trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
                framework="torch",
                num_nodes=1,
                device=constants.UNKNOWN,
                device_count=constants.UNKNOWN,
            ),
        ),
        ml_policy=TrainerV1alpha1MLPolicy(
            torch=TrainerV1alpha1TorchMLPolicySource(
                elasticPolicy=TrainerV1alpha1TorchElasticPolicy(
                    maxNodes=1, minNodes=1, maxRestarts=1
                ),
                numProcPerNode=IoK8sApimachineryPkgUtilIntstrIntOrString(1),
            )
        ),
    )
]


# Create venv script


# The exec script to embed training function into container command.
DEPENDENCIES_SCRIPT = textwrap.dedent(
    """
        PIP_DISABLE_PIP_VERSION_CHECK=1 pip install $QUIET \
    --no-warn-script-location $PIP_INDEX $PACKAGE_STR
    """
)

# activate virtualenv, then run the entrypoint from the virtualenv bin
LOCAL_EXEC_ENTRYPOINT = textwrap.dedent(
    """
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


LOCAL_EXEC_JOB_TEMPLATE = textwrap.dedent(
    """
    $OS_PYTHON_BIN -m venv --without-pip $PYENV_LOCATION
    echo "Operating inside $PYENV_LOCATION"
    source $PYENV_LOCATION/bin/activate
    $PYENV_LOCATION/bin/python -m ensurepip --upgrade --default-pip
    $DEPENDENCIES_SCRIPT
    $ENTRYPOINT
    $CLEANUP_SCRIPT
    """
)

LOCAL_EXEC_FILENAME = "train_{}.py"
