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

from kubeflow.trainer.backends.k8s import K8SBackend
from kubeflow.trainer.backends.local_process import LocalProcessBackend
from kubeflow.trainer.types.backends import K8SBackendConfig, LocalProcessBackendConfig

TRAINER_BACKEND_REGISTRY = {
    "kubernetes": {
        "backend_cls": K8SBackend,
        "config_cls": K8SBackendConfig,
    },
    "local": {
        "backend_cls": LocalProcessBackend,
        "config_cls": LocalProcessBackendConfig,
    }
}