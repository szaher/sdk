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

import threading
import subprocess
import logging
from datetime import datetime
from typing import List, Union

from kubeflow.trainer.local import resource_manager

logger = logging.getLogger(__name__)


class LocalJob(threading.Thread):
    def __init__(
            self, name, command: Union[List, str], mem_limit=None,
            cpu_time=None, cpu_limit=None, nice=0
    ):
        """Create a LocalJob. Create a local subprocess with threading to allow users
        to create background jobs.
        :param name: The name of the job.
        :type name: str
        :param command: The command to run.
        :type command: str
        """
        super().__init__()
        self.name = name
        self.command = command
        self._stdout = ""
        self._returncode = None
        self._success = False
        self._lock = threading.Lock()
        self._process = None
        self._output_updated = threading.Event()
        self._cancel_requested = threading.Event()
        self._start_time = None
        self._end_time = None
        # limit cpu and memory resources
        self.__memory_limit = mem_limit
        self.__cpu_time = cpu_time
        self.__cpu_limit = cpu_limit
        self.__nice = nice

    def run(self):
        logger.debug(f"[{self.name}] Starting...")
        try:
            self._start_time = datetime.now()
            self._process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                # @szaher how do we need to handle signals passed to child processes?
                preexec_fn=lambda: resource_manager.setup_local_process(
                    mem_limit=self.__memory_limit,
                    cpu_time=self.__cpu_time,
                    cpu_limit=self.__cpu_limit,
                    nice=self.__nice,
                )
            )

            while True:
                if self._cancel_requested.is_set():
                    self._process.terminate()
                    self._stdout += "[TrainingCancelled]\n"
                    self._success = False
                    return

                # Read output line by line (for streaming)
                output_line = self._process.stdout.readline()
                with self._lock:
                    if output_line:
                        self._stdout += output_line
                        self._output_updated.set()

                if not output_line and self._process.poll() is not None:
                    break

            self._process.stdout.close()
            self._returncode = self._process.wait()
            self._end_time = datetime.now()
            self._success = (self._process.returncode == 0)
            msg = (f"[{self.name}] Completed with code {self._returncode}"
                   f" in {self._end_time - self._start_time} seconds.")
            self._stdout += msg

        except Exception as e:
            with self._lock:
                self._stdout += f"Exception: {e}\n"
                self._success = False

    @property
    def stdout(self):
        with self._lock:
            return self._stdout

    @property
    def success(self):
        return self._success

    def cancel(self):
        self._cancel_requested.set()

    @property
    def returncode(self):
        return self._returncode

    def logs(self, follow=False) -> List[str]:
        """Print log lines"""
        if not follow:
            return self._stdout.splitlines()
        output_lines = ""
        try:
            for line in next(self.__follow_logs()):
                print(line, end="")
                output_lines += line
        except StopIteration:
            pass

        return output_lines.splitlines()

    def __follow_logs(self):
        """Generator that yields new output lines as they come in."""
        last_index = 0
        while self.is_alive() or last_index < len(self._stdout):
            self._output_updated.wait(timeout=1)
            with self._lock:
                data = self._stdout
                new_data = data[last_index:]
                last_index = len(data)
                self._output_updated.clear()
            if new_data:
                yield new_data

    @property
    def creation_time(self):
        return self._start_time

    @property
    def completion_time(self):
        return self._end_time
