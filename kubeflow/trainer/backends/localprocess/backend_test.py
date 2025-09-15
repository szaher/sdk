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

"""
Unit tests for the LocalProcessBackend class in the Kubeflow Trainer SDK.

This module uses pytest and unittest.mock to test LocalProcessBackend's behavior
across job creation, management, and lifecycle operations using local subprocess execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Type
from unittest.mock import Mock, patch

import pytest

from kubeflow.trainer.backends.localprocess.backend import LocalProcessBackend
from kubeflow.trainer.backends.localprocess.constants import local_runtimes
from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.backends.localprocess.types import (
    LocalBackendJobs,
    LocalBackendStep,
    LocalProcessBackendConfig,
)
from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types


@dataclass
class TestCase:
    """Test case configuration for parametrized tests."""

    name: str
    expected_status: str
    config: dict[str, Any] = field(default_factory=dict)
    expected_output: Optional[Any] = None
    expected_error: Optional[Type[Exception]] = None
    __test__ = False


# --------------------------
# Constants for test scenarios
# --------------------------
SUCCESS = "success"
FAILED = "failed"
TIMEOUT = "timeout"
RUNTIME_ERROR = "runtime_error"
VALUE_ERROR = "value_error"
TORCH_RUNTIME = constants.TORCH_RUNTIME
BASIC_TRAIN_JOB_NAME = "basic-job"
TEST_VENV_DIR = "/tmp/test_venv"


# --------------------------
# Fixtures
# --------------------------


@pytest.fixture
def local_backend():
    """Provide a LocalProcessBackend instance for testing."""
    config = LocalProcessBackendConfig(cleanup_venv=True)
    return LocalProcessBackend(cfg=config)


# --------------------------
# Object Creators
# --------------------------


def create_mock_runtime(
    name: str = TORCH_RUNTIME,
    framework: str = "torch",
    trainer_type: types.TrainerType = types.TrainerType.CUSTOM_TRAINER,
) -> types.Runtime:
    """Create a mock Runtime object for testing."""
    return types.Runtime(
        name=name,
        trainer=types.RuntimeTrainer(
            trainer_type=trainer_type,
            framework=framework,
            num_nodes=1,
            device=constants.UNKNOWN,
            device_count=constants.UNKNOWN,
        ),
        pretrained_model=None,
    )


def create_mock_trainer(
    func_name: str = "Training function",
    packages: Optional[list[str]] = None,
    env: Optional[dict[str, str]] = None,
) -> types.CustomTrainer:
    """Create a mock CustomTrainer object for testing."""
    if packages is None:
        packages = ["torch", "numpy"]
    if env is None:
        env = {"ENV_VAR": "test_value"}

    return types.CustomTrainer(
        func=lambda: print(func_name),
        func_args={"param1": "value1"},
        packages_to_install=packages,
        env=env,
    )


def create_mock_builtin_trainer() -> types.BuiltinTrainer:
    """Create a mock BuiltinTrainer object for testing."""
    return types.BuiltinTrainer(config=types.TorchTuneConfig())


def create_local_job(
    name: str = "test-job",
    status: str = constants.TRAINJOB_RUNNING,
) -> LocalJob:
    """Create a mock LocalJob object for testing."""
    mock_job = Mock(spec=LocalJob)
    mock_job.name = name
    mock_job.status = status
    mock_job.logs.return_value = ["log line 1", "log line 2"]
    mock_job.cancel = Mock()
    mock_job.join = Mock()
    return mock_job


def create_local_backend_job(
    name: str = "test-job",
    runtime: Optional[types.Runtime] = None,
    steps: Optional[list[LocalBackendStep]] = None,
) -> LocalBackendJobs:
    """Create a mock LocalBackendJobs object for testing."""
    if runtime is None:
        runtime = create_mock_runtime()
    if steps is None:
        mock_job = create_local_job()
        steps = [LocalBackendStep(step_name="train", job=mock_job)]

    return LocalBackendJobs(
        name=name,
        runtime=runtime,
        created=datetime.now(),
        steps=steps,
    )


def create_train_job_type(
    name: str = BASIC_TRAIN_JOB_NAME,
    runtime: Optional[types.Runtime] = None,
    status: str = constants.TRAINJOB_COMPLETE,
) -> types.TrainJob:
    """Create a mock TrainJob object for testing."""
    if runtime is None:
        runtime = create_mock_runtime()

    return types.TrainJob(
        name=name,
        creation_timestamp=datetime.now(),
        runtime=runtime,
        steps=[
            types.Step(
                name="train",
                status=status,
                pod_name="train-pod",
                device=constants.UNKNOWN,
                device_count=constants.UNKNOWN,
            )
        ],
        num_nodes=1,
        status=status,
    )


# --------------------------
# Tests
# --------------------------


def test_init():
    """Test LocalProcessBackend initialization."""
    config = LocalProcessBackendConfig(cleanup_venv=False)
    backend = LocalProcessBackend(cfg=config)

    assert backend.cfg == config
    assert not backend.cfg.cleanup_venv
    assert len(backend._LocalProcessBackend__local_jobs) == 0


def test_list_runtimes(local_backend):
    """Test list_runtimes method."""
    runtimes = local_backend.list_runtimes()

    assert isinstance(runtimes, list)
    assert len(runtimes) == len(local_runtimes)

    # Check that all returned items are Runtime objects
    for runtime in runtimes:
        assert isinstance(runtime, types.Runtime)
        assert isinstance(runtime.trainer, types.RuntimeTrainer)


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid runtime name",
            expected_status=SUCCESS,
            config={"name": TORCH_RUNTIME},
        ),
        TestCase(
            name="invalid runtime name",
            expected_status=FAILED,
            config={"name": "invalid-runtime"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_runtime(local_backend, test_case):
    """Test get_runtime method with various scenarios."""
    print("Executing test:", test_case.name)
    try:
        runtime = local_backend.get_runtime(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert isinstance(runtime, types.Runtime)
        assert runtime.name == test_case.config["name"]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
        if "invalid" in test_case.config.get("name", ""):
            assert "not found" in str(e)
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="valid runtime packages",
            expected_status=SUCCESS,
            config={"runtime": create_mock_runtime()},
        ),
        TestCase(
            name="invalid runtime packages",
            expected_status=FAILED,
            config={
                "runtime": types.Runtime(
                    name="invalid-runtime",
                    trainer=types.RuntimeTrainer(
                        trainer_type=types.TrainerType.CUSTOM_TRAINER,
                        framework="invalid",
                    ),
                )
            },
            expected_error=ValueError,
        ),
    ],
)
def test_get_runtime_packages(local_backend, test_case):
    """Test get_runtime_packages method with various scenarios."""
    print("Executing test:", test_case.name)
    try:
        packages = local_backend.get_runtime_packages(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert isinstance(packages, list)
        assert "torch" in packages

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
        assert "not found" in str(e)
    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="successful train with custom trainer",
            expected_status=SUCCESS,
            config={
                "runtime": create_mock_runtime(),
                "trainer": create_mock_trainer(),
            },
        ),
        TestCase(
            name="failed train with builtin trainer",
            expected_status=FAILED,
            config={
                "runtime": create_mock_runtime(),
                "trainer": create_mock_builtin_trainer(),
            },
            expected_error=ValueError,
        ),
    ],
)
@patch("kubeflow.trainer.backends.localprocess.backend.tempfile.mkdtemp")
@patch("kubeflow.trainer.backends.localprocess.backend.local_utils")
@patch("kubeflow.trainer.backends.localprocess.backend.LocalJob")
@patch("uuid.uuid4")
@patch("random.choice")
def test_train(
    mock_random_choice,
    mock_uuid,
    mock_local_job_class,
    mock_local_utils,
    mock_mkdtemp,
    local_backend,
    test_case,
):
    """Test train method with various scenarios."""
    print("Executing test:", test_case.name)

    # Setup mocks
    mock_random_choice.return_value = "a"
    mock_uuid.return_value.hex = "mock-uuid-hex"
    mock_mkdtemp.return_value = TEST_VENV_DIR

    mock_local_job = create_local_job(name="amock-uuid-h-train")
    mock_local_job_class.return_value = mock_local_job

    mock_local_utils.get_local_runtime_trainer.return_value = Mock()
    mock_local_utils.get_local_train_job_script.return_value = ["python", "script.py"]

    try:
        job_name = local_backend.train(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert job_name == "amock-uuid-h"
        assert len(local_backend._LocalProcessBackend__local_jobs) == 1

        # Verify mock calls
        mock_mkdtemp.assert_called_once()
        mock_local_utils.get_local_runtime_trainer.assert_called_once()
        mock_local_utils.get_local_train_job_script.assert_called_once()
        mock_local_job.start.assert_called_once()

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
        assert "CustomTrainer must be set" in str(e)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="list jobs when empty",
            expected_status=SUCCESS,
            config={},
            expected_output=[],
        ),
        TestCase(
            name="list jobs with existing jobs",
            expected_status=SUCCESS,
            config={"setup_jobs": True},
        ),
        TestCase(
            name="list jobs with runtime filter",
            expected_status=SUCCESS,
            config={"setup_jobs": True, "runtime": create_mock_runtime()},
        ),
    ],
)
def test_list_jobs(local_backend, test_case):
    """Test list_jobs method with various scenarios."""
    print("Executing test:", test_case.name)

    # Setup jobs if requested
    if test_case.config.get("setup_jobs"):
        backend_job = create_local_backend_job()
        local_backend._LocalProcessBackend__local_jobs.append(backend_job)

    try:
        runtime_filter = test_case.config.get("runtime")
        jobs = local_backend.list_jobs(runtime=runtime_filter)

        assert test_case.expected_status == SUCCESS
        assert isinstance(jobs, list)

        if test_case.config.get("setup_jobs"):
            assert len(jobs) >= 1
            for job in jobs:
                assert isinstance(job, types.TrainJob)
        else:
            assert jobs == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get existing job",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME},
        ),
        TestCase(
            name="get non-existent job",
            expected_status=FAILED,
            config={"name": "non-existent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_job(local_backend, test_case):
    """Test get_job method with various scenarios."""
    print("Executing test:", test_case.name)

    # Setup a job if testing success case
    if test_case.expected_status == SUCCESS:
        backend_job = create_local_backend_job(name=test_case.config["name"])
        local_backend._LocalProcessBackend__local_jobs.append(backend_job)

    try:
        with patch.object(
            local_backend,
            "_LocalProcessBackend__get_job_status",
            return_value=constants.TRAINJOB_COMPLETE,
        ):
            job = local_backend.get_job(**test_case.config)

        assert test_case.expected_status == SUCCESS
        assert isinstance(job, types.TrainJob)
        assert job.name == test_case.config["name"]

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
        assert "No TrainJob with name" in str(e)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="get logs for existing job",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME},
            expected_output=["log line 1", "log line 2"],
        ),
        TestCase(
            name="get logs with follow enabled",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME, "follow": True},
            expected_output=["log line 1", "log line 2"],
        ),
        TestCase(
            name="get logs for specific step",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME, "step": "train"},
            expected_output=["log line 1", "log line 2"],
        ),
        TestCase(
            name="get logs for non-existent job",
            expected_status=FAILED,
            config={"name": "non-existent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_get_job_logs(local_backend, test_case):
    """Test get_job_logs method with various scenarios."""
    print("Executing test:", test_case.name)

    # Setup a job if testing success case
    if test_case.expected_status == SUCCESS:
        backend_job = create_local_backend_job(name=test_case.config["name"])
        local_backend._LocalProcessBackend__local_jobs.append(backend_job)

    try:
        logs = list(local_backend.get_job_logs(**test_case.config))

        assert test_case.expected_status == SUCCESS
        assert logs == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
        assert "No TrainJob with name" in str(e)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="wait for job with default status",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME},
        ),
        TestCase(
            name="wait for job with custom status",
            expected_status=SUCCESS,
            config={
                "name": BASIC_TRAIN_JOB_NAME,
                "status": {constants.TRAINJOB_RUNNING, constants.TRAINJOB_COMPLETE},
                "timeout": 300,
                "polling_interval": 5,
            },
        ),
        TestCase(
            name="wait for non-existent job",
            expected_status=FAILED,
            config={"name": "non-existent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_wait_for_job_status(local_backend, test_case):
    """Test wait_for_job_status method with various scenarios."""
    print("Executing test:", test_case.name)

    # Setup a job if testing success case
    if test_case.expected_status == SUCCESS:
        mock_job = create_local_job(status=constants.TRAINJOB_RUNNING)
        backend_job = create_local_backend_job(
            name=test_case.config["name"],
            steps=[LocalBackendStep(step_name="train", job=mock_job)],
        )
        local_backend._LocalProcessBackend__local_jobs.append(backend_job)

    try:
        with patch.object(local_backend, "get_job") as mock_get_job:
            mock_train_job = create_train_job_type(name=test_case.config["name"])
            mock_get_job.return_value = mock_train_job

            result = local_backend.wait_for_job_status(**test_case.config)

            assert test_case.expected_status == SUCCESS
            assert result == mock_train_job
            mock_get_job.assert_called_once_with(test_case.config["name"])

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
        assert "No TrainJob with name" in str(e)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="delete existing job",
            expected_status=SUCCESS,
            config={"name": BASIC_TRAIN_JOB_NAME},
        ),
        TestCase(
            name="delete non-existent job",
            expected_status=FAILED,
            config={"name": "non-existent-job"},
            expected_error=ValueError,
        ),
    ],
)
def test_delete_job(local_backend, test_case):
    """Test delete_job method with various scenarios."""
    print("Executing test:", test_case.name)

    # Setup a job if testing success case
    if test_case.expected_status == SUCCESS:
        backend_job = create_local_backend_job(name=test_case.config["name"])
        local_backend._LocalProcessBackend__local_jobs.append(backend_job)
        initial_count = len(local_backend._LocalProcessBackend__local_jobs)

    try:
        local_backend.delete_job(**test_case.config)

        assert test_case.expected_status == SUCCESS
        # Verify job was removed
        assert len(local_backend._LocalProcessBackend__local_jobs) == initial_count - 1

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error
        assert "No TrainJob with name" in str(e)

    print("test execution complete")


@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(
            name="job status with failed step",
            expected_status=SUCCESS,
            config={"statuses": [constants.TRAINJOB_FAILED, constants.TRAINJOB_COMPLETE]},
            expected_output=constants.TRAINJOB_FAILED,
        ),
        TestCase(
            name="job status with running step",
            expected_status=SUCCESS,
            config={"statuses": [constants.TRAINJOB_RUNNING, constants.TRAINJOB_COMPLETE]},
            expected_output=constants.TRAINJOB_RUNNING,
        ),
        TestCase(
            name="job status with created step",
            expected_status=SUCCESS,
            config={"statuses": [constants.TRAINJOB_CREATED]},
            expected_output=constants.TRAINJOB_CREATED,
        ),
        TestCase(
            name="job status with unknown step",
            expected_status=SUCCESS,
            config={"statuses": ["unknown_status"]},
            expected_output=constants.TRAINJOB_CREATED,
        ),
    ],
)
def test_get_job_status(local_backend, test_case):
    """Test private __get_job_status method with various scenarios."""
    print("Executing test:", test_case.name)

    # Create mock steps with the specified statuses
    steps = []
    for status in test_case.config["statuses"]:
        mock_job = create_local_job(status=status)
        steps.append(LocalBackendStep(step_name=f"step-{status}", job=mock_job))

    backend_job = create_local_backend_job(steps=steps)

    try:
        status = local_backend._LocalProcessBackend__get_job_status(backend_job)

        assert test_case.expected_status == SUCCESS
        assert status == test_case.expected_output

    except Exception as e:
        assert test_case.expected_status == FAILED
        assert type(e) is test_case.expected_error

    print("test execution complete")


def test_register_job_scenarios(local_backend):
    """Test __register_job method with various scenarios."""
    print("Testing job registration scenarios")

    mock_job1 = create_local_job(name="job1")
    mock_job2 = create_local_job(name="job2")
    runtime = create_mock_runtime()

    # Test new job registration
    local_backend._LocalProcessBackend__register_job(
        train_job_name="new-job",
        step_name="train",
        job=mock_job1,
        runtime=runtime,
    )

    assert len(local_backend._LocalProcessBackend__local_jobs) == 1
    registered_job = local_backend._LocalProcessBackend__local_jobs[0]
    assert registered_job.name == "new-job"
    assert len(registered_job.steps) == 1
    assert registered_job.steps[0].step_name == "train"

    # Test adding step to existing job
    local_backend._LocalProcessBackend__register_job(
        train_job_name="new-job",
        step_name="validate",
        job=mock_job2,
        runtime=runtime,
    )

    assert len(local_backend._LocalProcessBackend__local_jobs) == 1
    assert len(registered_job.steps) == 2

    # Test duplicate step warning
    with patch("kubeflow.trainer.backends.localprocess.backend.logger") as mock_logger:
        local_backend._LocalProcessBackend__register_job(
            train_job_name="new-job",
            step_name="train",
            job=mock_job1,
            runtime=runtime,
        )
        mock_logger.warning.assert_called_once_with("Step 'train' already registered.")

    print("Job registration tests complete")


def test_convert_local_runtime_to_runtime(local_backend):
    """Test __convert_local_runtime_to_runtime method."""
    print("Testing runtime conversion")

    # Use the first local runtime from constants
    local_runtime = local_runtimes[0]

    converted_runtime = local_backend._LocalProcessBackend__convert_local_runtime_to_runtime(
        local_runtime
    )

    assert isinstance(converted_runtime, types.Runtime)
    assert converted_runtime.name == local_runtime.name
    assert converted_runtime.trainer.framework == local_runtime.trainer.framework
    assert converted_runtime.trainer.num_nodes == local_runtime.trainer.num_nodes
    assert converted_runtime.trainer.device == local_runtime.trainer.device
    assert converted_runtime.trainer.device_count == local_runtime.trainer.device_count
    assert converted_runtime.pretrained_model == local_runtime.pretrained_model

    print("Runtime conversion test complete")
