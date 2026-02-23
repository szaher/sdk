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

"""Runtime code for LiveTrainer hot-reloading.

Contains the self-contained instrumentation function that is extracted via
inspect.getsource() and injected into training pods, plus the wrapper generator.
"""

import inspect
import textwrap

from kubeflow.trainer.livetrainer.types import LiveTrainer, SyncConfig


def _create_live_trainer_instrumentation(
    config_file_path: str,
    sync_interval: int,
    cooldown_seconds: float,
    use_lock_file: bool,
    lock_file_path: str,
    checksum_validation: bool,
    hot_reload_params: list,
) -> tuple:
    """Self-contained instrumentation for LiveTrainer hot-reloading.

    This function is NOT called directly in the SDK - it's extracted as source code
    via inspect.getsource() and injected into user training scripts. This approach
    provides syntax highlighting, testability, and type checking while avoiding
    runtime SDK dependencies.

    Args:
        config_file_path: Full path to the YAML config file.
        sync_interval: Steps between sync checks.
        cooldown_seconds: Seconds to wait after file change detection.
        use_lock_file: Whether to use lock file protocol.
        lock_file_path: Full path to the lock file.
        checksum_validation: Whether to validate with MD5 checksum.
        hot_reload_params: List of allowed param names (empty list = all allowed).

    Returns:
        Tuple of (sync_loop, apply_fn) for use in training code.
    """
    import hashlib
    import json
    import logging
    import os
    import time

    import yaml

    logger = logging.getLogger("kubeflow.livetrainer")

    class FileWatcher:
        """Polls a YAML config file on NFS for changes using mtime + optional checksum."""

        def __init__(
            self,
            config_file_path: str,
            cooldown_seconds: float,
            use_lock_file: bool,
            lock_file_path: str,
            checksum_validation: bool,
        ):
            self._config_file_path = config_file_path
            self._cooldown_seconds = cooldown_seconds
            self._use_lock_file = use_lock_file
            self._lock_file_path = lock_file_path
            self._checksum_validation = checksum_validation
            self._last_mtime: float = 0.0
            self._last_checksum: str = ""
            self._last_params: dict = None

        def _compute_checksum(self, data: bytes) -> str:
            """Compute MD5 checksum of raw file data."""
            return hashlib.md5(data).hexdigest()

        def check_for_updates(self) -> dict:
            """Check if the config file has changed.

            Returns:
                Updated params dict, or None if no change detected.
            """
            if not os.path.exists(self._config_file_path):
                return None

            # If lock file protocol is enabled, skip if lock file exists
            if self._use_lock_file and os.path.exists(self._lock_file_path):
                return None

            try:
                current_mtime = os.path.getmtime(self._config_file_path)
            except OSError:
                return None

            if current_mtime <= self._last_mtime:
                return None

            # Cooldown to avoid reading partial NFS writes
            if self._cooldown_seconds > 0:
                time.sleep(self._cooldown_seconds)

            # Re-check lock file after cooldown
            if self._use_lock_file and os.path.exists(self._lock_file_path):
                return None

            try:
                with open(self._config_file_path, "rb") as f:
                    raw_data = f.read()
            except OSError as e:
                logger.warning("[LiveTrainer] Failed to read config file: %s", e)
                return None

            # Checksum validation
            if self._checksum_validation:
                current_checksum = self._compute_checksum(raw_data)
                if current_checksum == self._last_checksum:
                    self._last_mtime = current_mtime
                    return None

            # Parse YAML
            try:
                params = yaml.safe_load(raw_data)
            except yaml.YAMLError as e:
                logger.warning(
                    "[LiveTrainer] YAML parse error in %s: %s. "
                    "Continuing with previous parameters.",
                    self._config_file_path,
                    e,
                )
                return None

            if not isinstance(params, dict):
                logger.warning(
                    "[LiveTrainer] Config file must contain a YAML mapping. "
                    "Continuing with previous parameters."
                )
                return None

            # Update tracking state
            self._last_mtime = current_mtime
            if self._checksum_validation:
                self._last_checksum = self._compute_checksum(raw_data)
            self._last_params = params

            return params

    class SyncControlLoop:
        """Manages step-based sync checks with distributed broadcast support."""

        def __init__(
            self,
            file_watcher: FileWatcher,
            sync_interval: int,
            hot_reload_params: list,
        ):
            self._file_watcher = file_watcher
            self._sync_interval = sync_interval
            self._hot_reload_params = hot_reload_params
            self._step_count: int = 0
            self._distributed_initialized: bool = False
            self._rank: int = 0

            # Check if torch.distributed is available and initialized
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    self._distributed_initialized = True
                    self._rank = dist.get_rank()
            except (ImportError, RuntimeError):
                pass

        def should_sync(self) -> bool:
            """Check if current step should trigger a sync check."""
            self._step_count += 1
            return self._step_count % self._sync_interval == 0

        def _filter_params(self, params: dict) -> dict:
            """Filter params by whitelist if configured."""
            if not self._hot_reload_params:
                return params
            return {k: v for k, v in params.items() if k in self._hot_reload_params}

        def sync_params(self) -> dict:
            """Sync parameters across all ranks.

            Rank 0 checks FileWatcher; broadcasts updates to all workers via
            ``torch.distributed.broadcast``. In single-process mode, reads directly
            from FileWatcher without requiring torch.

            Returns:
                Filtered param dict, or None if no update available.
            """
            if not self._distributed_initialized:
                # Single-process mode: read directly (no torch needed)
                params = self._file_watcher.check_for_updates()
                if params is None:
                    return None
                return self._filter_params(params)

            import torch
            import torch.distributed as dist

            # Rank 0 checks for updates
            payload_bytes = b""
            if self._rank == 0:
                params = self._file_watcher.check_for_updates()
                if params is not None:
                    filtered = self._filter_params(params)
                    if filtered:
                        payload_bytes = json.dumps(filtered).encode("utf-8")

            # Broadcast payload length
            length_tensor = torch.tensor([len(payload_bytes)], dtype=torch.long, device="cpu")
            dist.broadcast(length_tensor, src=0)
            payload_length = length_tensor.item()

            if payload_length == 0:
                return None

            # Broadcast payload data
            if self._rank == 0:
                data_tensor = torch.tensor(list(payload_bytes), dtype=torch.uint8, device="cpu")
            else:
                data_tensor = torch.zeros(payload_length, dtype=torch.uint8, device="cpu")

            dist.broadcast(data_tensor, src=0)

            # Decode
            decoded_bytes = bytes(data_tensor.tolist())
            updates = json.loads(decoded_bytes.decode("utf-8"))
            return updates

    def apply_param_updates(optimizer: object, updates: dict) -> None:
        """Apply parameter updates to optimizer param groups.

        Maps common parameter names to PyTorch optimizer param group keys:
        - learning_rate -> lr
        - Other keys are applied directly if they exist in param groups.

        Args:
            optimizer: PyTorch optimizer instance.
            updates: Dict of parameter name to new value.
        """
        # Map friendly names to optimizer param group keys
        key_map = {
            "learning_rate": "lr",
        }

        for param_name, value in updates.items():
            opt_key = key_map.get(param_name, param_name)
            applied = False
            for group in optimizer.param_groups:
                if opt_key in group:
                    group[opt_key] = value
                    applied = True
            if applied:
                logger.info("[LiveTrainer] Updated %s = %s", opt_key, value)
            else:
                logger.warning(
                    "[LiveTrainer] Parameter '%s' not found in optimizer param groups",
                    opt_key,
                )

    # Create instances
    watcher = FileWatcher(
        config_file_path=config_file_path,
        cooldown_seconds=cooldown_seconds,
        use_lock_file=use_lock_file,
        lock_file_path=lock_file_path,
        checksum_validation=checksum_validation,
    )

    sync_loop = SyncControlLoop(
        file_watcher=watcher,
        sync_interval=sync_interval,
        hot_reload_params=hot_reload_params,
    )

    return sync_loop, apply_param_updates


def get_live_trainer_instrumentation_wrapper(trainer: LiveTrainer) -> str:
    """Generate self-contained instrumentation wrapper via inspect.getsource.

    Extracts ``_create_live_trainer_instrumentation`` as source code and injects
    a call with the provided configuration parameters.

    Args:
        trainer: LiveTrainer instance with configuration.

    Returns:
        Python code as string with ``{{user_func_import_and_call}}`` placeholder.
    """
    sync_config = trainer.sync_config or SyncConfig()

    config_file_path = f"{trainer.shared_volume_mount_path}/{sync_config.config_file_name}"
    lock_file_path = f"{trainer.shared_volume_mount_path}/{sync_config.lock_file_name}"
    hot_reload_params = trainer.hot_reload_params or []

    # Extract the entire function source
    instrumentation_code = inspect.getsource(_create_live_trainer_instrumentation)
    instrumentation_code = textwrap.dedent(instrumentation_code)

    # Build the wrapper with function call
    wrapper = f"""# =============================================================================
# Kubeflow SDK - LiveTrainer Hot-Reload Instrumentation
# Generated by kubeflow.trainer.livetrainer.runtime
# =============================================================================

print("[LiveTrainer] Initializing hot-reload instrumentation", flush=True)

# Instrumentation function definition
{instrumentation_code}

# Initialize instrumentation
_kubeflow_sync_loop, _apply_param_updates = _create_live_trainer_instrumentation(
    config_file_path={config_file_path!r},
    sync_interval={sync_config.sync_interval!r},
    cooldown_seconds={sync_config.cooldown_seconds!r},
    use_lock_file={sync_config.use_lock_file!r},
    lock_file_path={lock_file_path!r},
    checksum_validation={sync_config.checksum_validation!r},
    hot_reload_params={hot_reload_params!r},
)
print("[LiveTrainer] Hot-reload instrumentation enabled", flush=True)

# =============================================================================
# USER TRAINING CODE
# =============================================================================

{{{{user_func_import_and_call}}}}"""

    return wrapper
