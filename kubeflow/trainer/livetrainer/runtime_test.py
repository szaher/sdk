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

"""Tests for LiveTrainer runtime code generation and instrumentation logic."""

import os
import tempfile

import pytest
import yaml

from kubeflow.trainer.livetrainer.runtime import (
    _create_live_trainer_instrumentation,
    get_live_trainer_instrumentation_wrapper,
)
from kubeflow.trainer.livetrainer.types import LiveTrainer, SyncConfig


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def dummy_train_fn():
    pass


class TestFileWatcher:
    """Tests for the FileWatcher class created by _create_live_trainer_instrumentation."""

    def test_no_config_file_returns_none(self):
        sync_loop, _ = _create_live_trainer_instrumentation(
            config_file_path="/nonexistent/params.yaml",
            sync_interval=1,
            cooldown_seconds=0,
            use_lock_file=False,
            lock_file_path="/nonexistent/params.yaml.lock",
            checksum_validation=False,
            hot_reload_params=[],
        )
        # Sync should return None when config file doesn't exist
        result = sync_loop.sync_params()
        assert result is None

    def test_detects_config_file_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "params.yaml")
            lock_path = os.path.join(tmpdir, "params.yaml.lock")

            # Create initial config
            with open(config_path, "w") as f:
                yaml.dump({"learning_rate": 0.01}, f)

            sync_loop, _ = _create_live_trainer_instrumentation(
                config_file_path=config_path,
                sync_interval=1,
                cooldown_seconds=0,
                use_lock_file=False,
                lock_file_path=lock_path,
                checksum_validation=False,
                hot_reload_params=[],
            )

            # First sync should detect the file
            result = sync_loop.sync_params()
            assert result == {"learning_rate": 0.01}

            # Second sync without changes should return None
            result = sync_loop.sync_params()
            assert result is None

    def test_detects_modified_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "params.yaml")
            lock_path = os.path.join(tmpdir, "params.yaml.lock")

            with open(config_path, "w") as f:
                yaml.dump({"learning_rate": 0.01}, f)

            sync_loop, _ = _create_live_trainer_instrumentation(
                config_file_path=config_path,
                sync_interval=1,
                cooldown_seconds=0,
                use_lock_file=False,
                lock_file_path=lock_path,
                checksum_validation=False,
                hot_reload_params=[],
            )

            # First read
            sync_loop.sync_params()

            # Modify the file (ensure mtime changes)
            import time

            time.sleep(0.05)
            with open(config_path, "w") as f:
                yaml.dump({"learning_rate": 0.001}, f)

            result = sync_loop.sync_params()
            assert result == {"learning_rate": 0.001}

    def test_lock_file_blocks_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "params.yaml")
            lock_path = os.path.join(tmpdir, "params.yaml.lock")

            with open(config_path, "w") as f:
                yaml.dump({"learning_rate": 0.01}, f)

            # Create lock file
            with open(lock_path, "w") as f:
                f.write("")

            sync_loop, _ = _create_live_trainer_instrumentation(
                config_file_path=config_path,
                sync_interval=1,
                cooldown_seconds=0,
                use_lock_file=True,
                lock_file_path=lock_path,
                checksum_validation=False,
                hot_reload_params=[],
            )

            # Should return None because lock file exists
            result = sync_loop.sync_params()
            assert result is None

            # Remove lock file
            os.remove(lock_path)

            # Now should detect the file
            result = sync_loop.sync_params()
            assert result == {"learning_rate": 0.01}

    def test_invalid_yaml_continues(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "params.yaml")
            lock_path = os.path.join(tmpdir, "params.yaml.lock")

            with open(config_path, "w") as f:
                f.write("invalid: yaml: content: [")

            sync_loop, _ = _create_live_trainer_instrumentation(
                config_file_path=config_path,
                sync_interval=1,
                cooldown_seconds=0,
                use_lock_file=False,
                lock_file_path=lock_path,
                checksum_validation=False,
                hot_reload_params=[],
            )

            # Should return None on parse error
            result = sync_loop.sync_params()
            assert result is None

    def test_checksum_validation_prevents_duplicate_reads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "params.yaml")
            lock_path = os.path.join(tmpdir, "params.yaml.lock")

            with open(config_path, "w") as f:
                yaml.dump({"learning_rate": 0.01}, f)

            sync_loop, _ = _create_live_trainer_instrumentation(
                config_file_path=config_path,
                sync_interval=1,
                cooldown_seconds=0,
                use_lock_file=False,
                lock_file_path=lock_path,
                checksum_validation=True,
                hot_reload_params=[],
            )

            # First read
            result = sync_loop.sync_params()
            assert result == {"learning_rate": 0.01}

            # Touch the file to update mtime, but don't change content
            import time

            time.sleep(0.05)
            os.utime(config_path, None)

            # Checksum should prevent re-reading same content
            result = sync_loop.sync_params()
            assert result is None


class TestSyncControlLoop:
    def test_should_sync_interval(self):
        sync_loop, _ = _create_live_trainer_instrumentation(
            config_file_path="/nonexistent/params.yaml",
            sync_interval=3,
            cooldown_seconds=0,
            use_lock_file=False,
            lock_file_path="/nonexistent/params.yaml.lock",
            checksum_validation=False,
            hot_reload_params=[],
        )

        # Steps 1, 2 should not sync; step 3 should
        assert sync_loop.should_sync() is False  # step 1
        assert sync_loop.should_sync() is False  # step 2
        assert sync_loop.should_sync() is True  # step 3
        assert sync_loop.should_sync() is False  # step 4
        assert sync_loop.should_sync() is False  # step 5
        assert sync_loop.should_sync() is True  # step 6

    def test_param_filtering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "params.yaml")
            lock_path = os.path.join(tmpdir, "params.yaml.lock")

            with open(config_path, "w") as f:
                yaml.dump(
                    {"learning_rate": 0.01, "momentum": 0.9, "secret": "value"},
                    f,
                )

            sync_loop, _ = _create_live_trainer_instrumentation(
                config_file_path=config_path,
                sync_interval=1,
                cooldown_seconds=0,
                use_lock_file=False,
                lock_file_path=lock_path,
                checksum_validation=False,
                hot_reload_params=["learning_rate", "momentum"],
            )

            result = sync_loop.sync_params()
            assert result == {"learning_rate": 0.01, "momentum": 0.9}
            assert "secret" not in result

    def test_empty_whitelist_allows_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "params.yaml")
            lock_path = os.path.join(tmpdir, "params.yaml.lock")

            with open(config_path, "w") as f:
                yaml.dump({"lr": 0.01, "momentum": 0.9}, f)

            sync_loop, _ = _create_live_trainer_instrumentation(
                config_file_path=config_path,
                sync_interval=1,
                cooldown_seconds=0,
                use_lock_file=False,
                lock_file_path=lock_path,
                checksum_validation=False,
                hot_reload_params=[],
            )

            result = sync_loop.sync_params()
            assert result == {"lr": 0.01, "momentum": 0.9}


@pytest.mark.skipif(not _torch_available(), reason="torch not installed")
class TestApplyParamUpdates:
    def test_apply_learning_rate(self):
        import torch

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        _, apply_fn = _create_live_trainer_instrumentation(
            config_file_path="/nonexistent/params.yaml",
            sync_interval=1,
            cooldown_seconds=0,
            use_lock_file=False,
            lock_file_path="/nonexistent/params.yaml.lock",
            checksum_validation=False,
            hot_reload_params=[],
        )

        apply_fn(optimizer, {"learning_rate": 0.001})
        assert optimizer.param_groups[0]["lr"] == 0.001

    def test_apply_direct_lr_key(self):
        import torch

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        _, apply_fn = _create_live_trainer_instrumentation(
            config_file_path="/nonexistent/params.yaml",
            sync_interval=1,
            cooldown_seconds=0,
            use_lock_file=False,
            lock_file_path="/nonexistent/params.yaml.lock",
            checksum_validation=False,
            hot_reload_params=[],
        )

        apply_fn(optimizer, {"lr": 0.005})
        assert optimizer.param_groups[0]["lr"] == 0.005

    def test_apply_momentum(self):
        import torch

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        _, apply_fn = _create_live_trainer_instrumentation(
            config_file_path="/nonexistent/params.yaml",
            sync_interval=1,
            cooldown_seconds=0,
            use_lock_file=False,
            lock_file_path="/nonexistent/params.yaml.lock",
            checksum_validation=False,
            hot_reload_params=[],
        )

        apply_fn(optimizer, {"momentum": 0.95})
        assert optimizer.param_groups[0]["momentum"] == 0.95

    def test_apply_weight_decay(self):
        import torch

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

        _, apply_fn = _create_live_trainer_instrumentation(
            config_file_path="/nonexistent/params.yaml",
            sync_interval=1,
            cooldown_seconds=0,
            use_lock_file=False,
            lock_file_path="/nonexistent/params.yaml.lock",
            checksum_validation=False,
            hot_reload_params=[],
        )

        apply_fn(optimizer, {"weight_decay": 1e-3})
        assert optimizer.param_groups[0]["weight_decay"] == 1e-3

    def test_unknown_param_does_not_crash(self):
        """Params not in optimizer should log warning but not crash."""
        import torch

        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        _, apply_fn = _create_live_trainer_instrumentation(
            config_file_path="/nonexistent/params.yaml",
            sync_interval=1,
            cooldown_seconds=0,
            use_lock_file=False,
            lock_file_path="/nonexistent/params.yaml.lock",
            checksum_validation=False,
            hot_reload_params=[],
        )

        # Should not raise
        apply_fn(optimizer, {"nonexistent_param": 42})
        # lr should be unchanged
        assert optimizer.param_groups[0]["lr"] == 0.01


class TestWrapperGeneration:
    def test_wrapper_compiles(self):
        trainer = LiveTrainer(
            func=dummy_train_fn,
            shared_volume_mount_path="/mnt/shared",
        )

        wrapper_code = get_live_trainer_instrumentation_wrapper(trainer)
        # Replace the placeholder with a simple statement
        code = wrapper_code.replace("{{user_func_import_and_call}}", "pass")

        # Should compile without errors
        compile(code, "<test>", "exec")

    def test_wrapper_contains_globals(self):
        trainer = LiveTrainer(
            func=dummy_train_fn,
            shared_volume_mount_path="/mnt/shared",
            sync_config=SyncConfig(sync_interval=20),
        )

        wrapper_code = get_live_trainer_instrumentation_wrapper(trainer)
        assert "_kubeflow_sync_loop" in wrapper_code
        assert "_apply_param_updates" in wrapper_code
        assert "sync_interval=20" in wrapper_code

    def test_wrapper_contains_config_path(self):
        trainer = LiveTrainer(
            func=dummy_train_fn,
            shared_volume_mount_path="/mnt/custom",
            sync_config=SyncConfig(config_file_name="hp.yaml"),
        )

        wrapper_code = get_live_trainer_instrumentation_wrapper(trainer)
        assert "/mnt/custom/hp.yaml" in wrapper_code

    def test_wrapper_contains_placeholder(self):
        trainer = LiveTrainer(func=dummy_train_fn)
        wrapper_code = get_live_trainer_instrumentation_wrapper(trainer)
        assert "{{user_func_import_and_call}}" in wrapper_code
