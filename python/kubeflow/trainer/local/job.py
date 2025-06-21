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

logger = logging.getLogger(__name__)


class LocalJob(threading.Thread):
    def __init__(self, name, command, dependencies=None):
        super().__init__()
        self.name = name
        self.command = command
        self.dependencies = dependencies or []
        self._stdout = ""
        self._stderr = ""
        self._returncode = None
        self._success = False
        self._lock = threading.Lock()
        self._output_updated = threading.Event()

    def run(self):
        for dep in self.dependencies:
            dep.join()
            if not dep.success:
                with self._lock:
                    self._stderr = f"Dependency {dep.name} failed. Skipping."
                return

        logger.debug(f"[{self.name}] Starting...")
        try:
            process = subprocess.Popen(
                self.command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            # Read output line by line (for streaming)
            for line in iter(process.stdout.readline, ''):
                with self._lock:
                    self._stdout += line
                self._output_updated.set()

            process.stdout.close()
            process.wait()

            with self._lock:
                self._returncode = process.returncode
                self._success = (process.returncode == 0)
            print(f"[{self.name}] Completed with code {self._returncode}.")
        except Exception as e:
            with self._lock:
                self._stderr += f"Exception: {e}\n"
                self._success = False

    @property
    def stdout(self):
        with self._lock:
            return self._stdout

    @property
    def success(self):
        return self._success

    @property
    def returncode(self):
        return self._returncode

    def follow_logs(self):
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
