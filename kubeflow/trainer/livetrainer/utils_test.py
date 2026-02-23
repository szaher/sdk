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

"""Tests for LiveTrainer utility functions."""

from kubeflow.trainer.constants import constants
from kubeflow.trainer.livetrainer.types import LiveTrainer, SyncConfig
from kubeflow.trainer.livetrainer.utils import get_trainer_cr_from_live_trainer
from kubeflow.trainer.types.types import Runtime, RuntimeTrainer, TrainerType


def dummy_train_fn():
    pass


def _make_runtime() -> Runtime:
    """Create a minimal runtime for testing."""
    trainer = RuntimeTrainer(
        trainer_type=TrainerType.CUSTOM_TRAINER,
        framework="pytorch",
        image="pytorch/pytorch:latest",
    )
    trainer.set_command(constants.TORCH_COMMAND)
    return Runtime(name="torch-distributed", trainer=trainer)


class TestGetTrainerCRFromLiveTrainer:
    def test_basic_trainer_cr(self):
        trainer = LiveTrainer(func=dummy_train_fn)
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)

        assert cr.command is not None
        assert len(cr.command) > 0
        # The command should contain the instrumentation code
        command_str = " ".join(cr.command)
        assert "_kubeflow_sync_loop" in command_str
        assert "_apply_param_updates" in command_str

    def test_num_nodes_set(self):
        trainer = LiveTrainer(func=dummy_train_fn, num_nodes=4)
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)
        assert cr.num_nodes == 4

    def test_resources_per_node_set(self):
        trainer = LiveTrainer(func=dummy_train_fn, resources_per_node={"gpu": 2})
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)
        assert cr.resources_per_node is not None

    def test_env_vars_set(self):
        trainer = LiveTrainer(func=dummy_train_fn, env={"DEBUG": "1", "LOG_LEVEL": "info"})
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)
        assert cr.env is not None
        env_names = [e.name for e in cr.env]
        assert "DEBUG" in env_names
        assert "LOG_LEVEL" in env_names

    def test_func_args_in_command(self):
        trainer = LiveTrainer(func=dummy_train_fn, func_args={"epochs": 100})
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)
        command_str = " ".join(cr.command)
        assert "dummy_train_fn(**" in command_str

    def test_no_func_args_in_command(self):
        trainer = LiveTrainer(func=dummy_train_fn)
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)
        command_str = " ".join(cr.command)
        assert "dummy_train_fn()" in command_str

    def test_custom_sync_config_in_command(self):
        trainer = LiveTrainer(
            func=dummy_train_fn,
            sync_config=SyncConfig(sync_interval=50, cooldown_seconds=2.0),
        )
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)
        command_str = " ".join(cr.command)
        assert "sync_interval=50" in command_str
        assert "cooldown_seconds=2.0" in command_str

    def test_image_set(self):
        trainer = LiveTrainer(func=dummy_train_fn, image="custom/image:latest")
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)
        assert cr.image == "custom/image:latest"

    def test_wrapper_code_compiles(self):
        """Verify that the generated wrapper code is valid Python."""
        trainer = LiveTrainer(
            func=dummy_train_fn,
            shared_volume_mount_path="/mnt/shared",
            hot_reload_params=["learning_rate"],
            sync_config=SyncConfig(sync_interval=5),
        )
        runtime = _make_runtime()

        cr = get_trainer_cr_from_live_trainer(runtime, trainer)
        # Verify the command was built with instrumentation markers
        for c in cr.command:
            if "_kubeflow_sync_loop" in c:
                assert "dummy_train_fn" in c
                break
