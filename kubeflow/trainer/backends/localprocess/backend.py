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
import logging
import os
import string
import tempfile
import uuid
import venv
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set, Union, Iterator

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types
from kubeflow.trainer.backends.base import ExecutionBackend
from kubeflow.trainer.backends.localprocess.types import (
    LocalProcessBackendConfig,
    LocalBackendJobs,
    LocalBackendStep,
)
from kubeflow.trainer.backends.localprocess.runtimes import local_runtimes
from kubeflow.trainer.backends.localprocess.job import LocalJob
from kubeflow.trainer.backends.localprocess import utils as local_utils

logger = logging.getLogger(__name__)


class LocalProcessBackend(ExecutionBackend):
    def __init__(
        self,
        cfg: LocalProcessBackendConfig,
    ):
        # list of running subprocesses
        self.__local_jobs: List[LocalBackendJobs] = []
        self.cfg = cfg

    def list_runtimes(self) -> List[types.Runtime]:
        return [local_runtime.runtime for local_runtime in local_runtimes]

    def get_runtime(self, name: str) -> Optional[types.Runtime]:
        _runtime = [rt.runtime for rt in local_runtimes if rt.runtime.name == name]
        if not _runtime:
            raise ValueError(f"Runtime '{name}' not found.")

        return _runtime[0]

    def get_runtime_packages(self, runtime: types.Runtime):
        raise NotImplementedError("get_runtime_packages is not supported by LocalProcessBackend")

    def train(
        self,
        runtime: Optional[types.Runtime] = None,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[Union[types.CustomTrainer, types.BuiltinTrainer]] = None,
    ) -> str:
        train_job_name = "kft-{}".format(
            random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11],
        )
        # Build the env
        if not trainer:
            raise ValueError("Cannot create TrainJob without a Trainer")
        if isinstance(trainer, types.CustomTrainer):
            trainer: types.CustomTrainer = trainer

        # setup runtime
        target_dir, python_bin, pip_bin = self.__setup_runtime(train_job_name=train_job_name)

        if self.cfg.debug:
            logger.info("operating in {}".format(target_dir))

        local_runtime = self.__get_full_runtime(runtime)

        runtime.trainer = local_utils.get_runtime_trainer(
            venv_dir=target_dir,
            python_bin=str(python_bin),
            framework=runtime.trainer.framework,
            ml_policy=local_runtime.ml_policy,
        )

        training_command = []
        deps_command = []

        if isinstance(trainer, types.CustomTrainer):
            if runtime.trainer.trainer_type != types.TrainerType.CUSTOM_TRAINER:
                raise ValueError(f"CustomTrainer can't be used with {runtime.name} runtime")
            if trainer.packages_to_install:
                deps_command = local_utils.get_dependencies_command(
                    python_bin=python_bin,
                    pip_bin=str(pip_bin),
                    pip_index_urls=trainer.pip_index_urls
                    if trainer.pip_index_urls
                    else constants.DEFAULT_PIP_INDEX_URLS,
                    packages=trainer.packages_to_install,
                )
            training_command = local_utils.get_command_using_train_func(
                runtime=runtime,
                train_func=trainer.func,
                train_func_parameters=trainer.func_args,
                venv_dir=target_dir,
                train_job_name=train_job_name,
            )
        # make sure we wait for dependencies to be installed and runtime to become ready
        training_dependencies = []
        # wait for all jobs to be completed then cleanup venv and other resources if needed.
        cleanup_dependencies = []

        if deps_command:
            deps_job = LocalJob(
                name="{}-deps".format(train_job_name),
                command=deps_command,
                debug=self.cfg.debug,
                execution_dir=target_dir,
                env=trainer.env,
            )
            deps_job.start()
            # make sure training doesn't start before dependencies installation finish
            training_dependencies.append(deps_job)
            self.__register_job(train_job_name, "deps", deps_job)

        if training_command:
            train_job = LocalJob(
                name="{}-train".format(train_job_name),
                command=training_command,
                debug=self.cfg.debug,
                execution_dir=target_dir,
                env=trainer.env,
                dependencies=training_dependencies,
            )
            train_job.start()
            # ask cleanup job to wait for training to be completed.
            cleanup_dependencies.append(train_job)
            self.__register_job(train_job_name, "train", train_job)

            # if cleanup is requested. The virtualenv dir will be deleted.
            if self.cfg.cleanup:
                cleanup_command = local_utils.get_cleanup_command(venv_dir=target_dir)
                cleanup_job = LocalJob(
                    name="{}-cleanup".format(train_job_name),
                    command=cleanup_command,
                    debug=self.cfg.debug,
                    execution_dir=target_dir,
                    env=trainer.env,
                    dependencies=cleanup_dependencies,
                )
                cleanup_job.start()
                self.__register_job(train_job_name, "cleanup", cleanup_job)

        return train_job_name

    def list_jobs(self, runtime: Optional[types.Runtime] = None) -> List[types.TrainJob]:
        result = [
            types.TrainJob(
                name=j.name,
                creation_timestamp=j.created,
                runtime=runtime,
                num_nodes=1,
                steps=[
                    types.Step(name=s.step_name, pod_name=s.step_name, status=s.job.status)
                    for s in j.steps
                ],
            )
            for j in self.__local_jobs
        ]

        return result

    def get_job(self, name: str) -> Optional[types.TrainJob]:
        _job = next((j for j in self.__local_jobs if j.name == name), None)
        if _job is None:
            raise ValueError("No TrainJob with name '%s'" % name)

        # check and set the correct job status to match `TrainerClient` supported statuses
        status = self.__get_job_status(_job[0])

        return types.TrainJob(
            name=_job[0].name,
            creation_timestamp=_job[0].created,
            steps=[
                types.Step(name=_step.step_name, pod_name=_step.step_name, status=_step.job.status)
                for _step in _job[0].steps
            ],
            runtime=None,
            num_nodes=1,
            status=status,
        )

    def get_job_logs(
        self,
        name: str,
        follow: Optional[bool] = False,
        step: str = constants.NODE + "-0",
        node_rank: int = 0,
    ) -> Iterator[str]:
        _job = [j for j in self.__local_jobs if j.name == name]
        if not _job:
            raise ValueError("No TrainJob with name '%s'" % name)

        want_all_steps = step == constants.NODE + "-0"

        for _step in _job[0].steps:
            if not want_all_steps and _step.step_name != step:
                continue
            # Flatten the generator and pass through flags so it behaves as expected
            # (adjust args if stream_logs has different signature)
            yield from _step.job.logs(follow=follow)

    def delete_job(self, name: str):
        # find job first.
        _job = next((j for j in self.__local_jobs if j.name == name), None)
        if _job is None:
            raise ValueError("No TrainJob with name '%s'" % name)

        # cancel all nested step jobs in target job
        _ = [step.job.cancel() for step in _job.steps]
        # remove the job from the list of jobs
        self.__local_jobs.remove(_job)

    def wait_for_job_status(
        self,
        name: str,
        status: Set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
        polling_interval: int = 2,
    ) -> types.TrainJob:
        # find first match or fallback
        _job = next((_job for _job in self.__local_jobs if _job.name == name), None)

        if _job is None:
            raise ValueError("No TrainJob with name '%s'" % name)
        # find a better implementation for this
        for _step in _job.steps:
            if _step.status in [constants.TRAINJOB_RUNNING, constants.TRAINJOB_CREATED]:
                _step.job.join(timeout=timeout)
        return self.get_job(name)

    def __setup_runtime(self, train_job_name):
        target_dir = tempfile.mkdtemp(prefix=f"{train_job_name}-")
        venv.create(env_dir=target_dir, with_pip=False)

        python_bin = Path(target_dir) / "bin" / "python"
        if not os.path.exists(python_bin):
            raise RuntimeError(f"Python executable not found at {python_bin}")
        pip_bin = Path(target_dir) / "bin" / "pip"

        return target_dir, python_bin, pip_bin

    def __get_full_runtime(self, runtime: types.Runtime):
        target_runtime = [rt for rt in local_runtimes if rt.runtime.name == runtime.name]
        if not target_runtime:
            raise ValueError(f"Runtime '{runtime.name}' not found.")
        return target_runtime[0]

    def __get_job_status(self, job: LocalBackendJobs) -> str:
        statuses = [_step.job.status for _step in job.steps]
        # if status is running or failed will take precedence over completed
        if constants.TRAINJOB_FAILED in statuses:
            status = constants.TRAINJOB_FAILED
        elif constants.TRAINJOB_RUNNING in statuses:
            status = constants.TRAINJOB_RUNNING
        elif constants.TRAINJOB_CREATED in statuses:
            status = constants.TRAINJOB_CREATED
        else:
            status = constants.TRAINJOB_CREATED

        return status

    def __register_job(self, train_job_name, step_name, job):
        _job = [j for j in self.__local_jobs if j.name == train_job_name]
        if not _job:
            _job = LocalBackendJobs(name=train_job_name, created=datetime.now())
            self.__local_jobs.append(_job)
        else:
            _job = _job[0]
        _step = [s for s in _job.steps if s.step_name == step_name]
        if not _step:
            _step = LocalBackendStep(step_name=step_name, job=job)
            _job.steps.append(_step)
        else:
            logger.warning("Step '{}' already registered.".format(step_name))
