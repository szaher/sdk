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

"""LiveTrainer and SyncConfig dataclass definitions."""

from collections.abc import Callable
from dataclasses import dataclass, field

from kubeflow.trainer.constants import constants
from kubeflow.trainer.livetrainer.constants import (
    DEFAULT_CONFIG_FILE_NAME,
    DEFAULT_LOCK_FILE_NAME,
    DEFAULT_SHARED_VOLUME_MOUNT_PATH,
)


@dataclass
class SyncConfig:
    """Configuration for the synchronization control loop.

    Args:
        sync_interval: Number of training steps between sync checks. Default: 10.
        cooldown_seconds: Seconds to wait after detecting a file change before reading,
            to avoid partial writes. Default: 0.5.
        config_file_name: Name of the YAML config file on the shared volume.
            Default: "params.yaml".
        use_lock_file: Whether to use a lock file for atomic write protocol. When True,
            the FileWatcher will wait until the lock file is removed before reading.
            Default: False.
        lock_file_name: Name of the lock file. Default: "params.yaml.lock".
        checksum_validation: Whether to validate file contents with MD5 checksum to
            detect partial writes. Default: True.

    Raises:
        ValueError: If sync_interval is not a positive integer.
        ValueError: If cooldown_seconds is negative.
        ValueError: If config_file_name is empty.
        ValueError: If lock_file_name is empty when use_lock_file is True.
    """

    sync_interval: int = 10
    cooldown_seconds: float = 0.5
    config_file_name: str = DEFAULT_CONFIG_FILE_NAME
    use_lock_file: bool = False
    lock_file_name: str = DEFAULT_LOCK_FILE_NAME
    checksum_validation: bool = True

    def __post_init__(self) -> None:
        """Validate SyncConfig fields."""
        if not isinstance(self.sync_interval, int) or self.sync_interval < 1:
            raise ValueError(f"sync_interval must be a positive integer, got {self.sync_interval}")

        if not isinstance(self.cooldown_seconds, (int, float)) or self.cooldown_seconds < 0:
            raise ValueError(
                f"cooldown_seconds must be a non-negative number, got {self.cooldown_seconds}"
            )

        if not self.config_file_name or not isinstance(self.config_file_name, str):
            raise ValueError("config_file_name must be a non-empty string")

        if self.use_lock_file and (
            not self.lock_file_name or not isinstance(self.lock_file_name, str)
        ):
            raise ValueError("lock_file_name must be a non-empty string when use_lock_file is True")


@dataclass
class LiveTrainer:
    """LiveTrainer configuration for hot-reloading hyperparameters during training.

    Enables real-time modification of optimizer parameters (learning rate, momentum,
    batch_size, etc.) by watching a YAML config file on a shared volume. Uses a
    polling-based FileWatcher and synchronized broadcast for distributed training.

    The shared volume (e.g. NFS, CephFS, or any RWX PVC) must be provisioned and
    mounted separately by the user or platform — for example, via PodTemplateOverrides
    or cluster-level configuration.

    Args:
        func: The training function that encapsulates the model training process.
            The function should call ``_kubeflow_sync_loop.should_sync()`` and
            ``_kubeflow_sync_loop.sync_params()`` to check for and apply parameter updates.
        shared_volume_mount_path: Local mount point for the shared volume in training pods.
            Default: "/mnt/shared".
        hot_reload_params: Whitelist of parameter names allowed for hot-reload.
            If None, all parameters in the config file are allowed.
        sync_config: Configuration for the synchronization control loop.
            If None, defaults are used.
        func_args: The arguments to pass to the training function as kwargs.
        image: Optional container image to use in the TrainJob.
        packages_to_install: A list of Python packages to install before running
            the training function.
        pip_index_urls: The PyPI URLs from which to install Python packages.
        num_nodes: The number of nodes to use for distributed training.
        resources_per_node: The computing resources to allocate per node.
        env: Environment variables to set in the training nodes.

    Raises:
        ValueError: If func is not callable.
        ValueError: If shared_volume_mount_path is empty.
        ValueError: If hot_reload_params is not a list of strings.
        ValueError: If sync_config is not a SyncConfig instance.
    """

    func: Callable
    shared_volume_mount_path: str = DEFAULT_SHARED_VOLUME_MOUNT_PATH
    hot_reload_params: list[str] | None = None
    sync_config: SyncConfig | None = None
    func_args: dict | None = None
    image: str | None = None
    packages_to_install: list[str] | None = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    num_nodes: int | None = None
    resources_per_node: dict | None = None
    env: dict[str, str] | None = None

    def __post_init__(self) -> None:
        """Validate LiveTrainer fields."""
        if not callable(self.func):
            raise ValueError(
                f"func must be callable, got {type(self.func).__name__}. "
                f"Please provide a training function."
            )

        if not isinstance(self.shared_volume_mount_path, str) or not self.shared_volume_mount_path:
            raise ValueError("shared_volume_mount_path must be a non-empty string")

        if self.hot_reload_params is not None:
            if not isinstance(self.hot_reload_params, list):
                raise ValueError("hot_reload_params must be a list of strings or None")
            if not all(isinstance(p, str) for p in self.hot_reload_params):
                raise ValueError("all items in hot_reload_params must be strings")

        if self.sync_config is not None and not isinstance(self.sync_config, SyncConfig):
            raise ValueError(
                f"sync_config must be a SyncConfig instance or None, "
                f"got {type(self.sync_config).__name__}"
            )
