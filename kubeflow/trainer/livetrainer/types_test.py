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

"""Tests for LiveTrainer and SyncConfig dataclass validation."""

import pytest

from kubeflow.trainer.livetrainer.types import LiveTrainer, SyncConfig


def dummy_train_fn():
    pass


# ---- SyncConfig validation tests ----


class TestSyncConfig:
    def test_defaults(self):
        config = SyncConfig()
        assert config.sync_interval == 10
        assert config.cooldown_seconds == 0.5
        assert config.config_file_name == "params.yaml"
        assert config.use_lock_file is False
        assert config.lock_file_name == "params.yaml.lock"
        assert config.checksum_validation is True

    def test_custom_values(self):
        config = SyncConfig(
            sync_interval=50,
            cooldown_seconds=2.0,
            config_file_name="hyperparams.yaml",
            use_lock_file=True,
            lock_file_name="hyperparams.yaml.lock",
            checksum_validation=False,
        )
        assert config.sync_interval == 50
        assert config.cooldown_seconds == 2.0
        assert config.config_file_name == "hyperparams.yaml"
        assert config.use_lock_file is True
        assert config.checksum_validation is False

    def test_invalid_sync_interval_zero(self):
        with pytest.raises(ValueError, match="sync_interval must be a positive integer"):
            SyncConfig(sync_interval=0)

    def test_invalid_sync_interval_negative(self):
        with pytest.raises(ValueError, match="sync_interval must be a positive integer"):
            SyncConfig(sync_interval=-5)

    def test_invalid_cooldown_negative(self):
        with pytest.raises(ValueError, match="cooldown_seconds must be a non-negative number"):
            SyncConfig(cooldown_seconds=-1.0)

    def test_cooldown_zero_is_valid(self):
        config = SyncConfig(cooldown_seconds=0)
        assert config.cooldown_seconds == 0

    def test_invalid_config_file_name_empty(self):
        with pytest.raises(ValueError, match="config_file_name must be a non-empty string"):
            SyncConfig(config_file_name="")

    def test_invalid_lock_file_name_when_enabled(self):
        with pytest.raises(ValueError, match="lock_file_name must be a non-empty string"):
            SyncConfig(use_lock_file=True, lock_file_name="")


# ---- LiveTrainer validation tests ----


class TestLiveTrainer:
    def test_minimal_valid(self):
        trainer = LiveTrainer(func=dummy_train_fn)
        assert trainer.func is dummy_train_fn
        assert trainer.shared_volume_mount_path == "/mnt/shared"
        assert trainer.hot_reload_params is None
        assert trainer.sync_config is None

    def test_full_config(self):
        sync_config = SyncConfig(sync_interval=20)
        trainer = LiveTrainer(
            func=dummy_train_fn,
            shared_volume_mount_path="/mnt/custom",
            hot_reload_params=["learning_rate", "batch_size"],
            sync_config=sync_config,
            func_args={"epochs": 100},
            num_nodes=4,
            resources_per_node={"gpu": 2},
            env={"DEBUG": "1"},
        )
        assert trainer.shared_volume_mount_path == "/mnt/custom"
        assert trainer.hot_reload_params == ["learning_rate", "batch_size"]
        assert trainer.sync_config.sync_interval == 20

    def test_invalid_func_not_callable(self):
        with pytest.raises(ValueError, match="func must be callable"):
            LiveTrainer(func="not_a_function")

    def test_invalid_shared_volume_mount_path_empty(self):
        with pytest.raises(ValueError, match="shared_volume_mount_path must be a non-empty string"):
            LiveTrainer(func=dummy_train_fn, shared_volume_mount_path="")

    def test_invalid_hot_reload_params_not_list(self):
        with pytest.raises(ValueError, match="hot_reload_params must be a list"):
            LiveTrainer(func=dummy_train_fn, hot_reload_params="learning_rate")

    def test_invalid_hot_reload_params_non_string_items(self):
        with pytest.raises(ValueError, match="all items in hot_reload_params must be strings"):
            LiveTrainer(func=dummy_train_fn, hot_reload_params=["lr", 123])

    def test_invalid_sync_config_type(self):
        with pytest.raises(ValueError, match="sync_config must be a SyncConfig instance"):
            LiveTrainer(func=dummy_train_fn, sync_config={"sync_interval": 10})
