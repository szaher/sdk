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
spawn_limited.py

Cross-platform subprocess launcher that can:
  • cap memory (bytes)
  • cap total CPU-seconds
  • pin to specific CPU cores
  • lower priority (nice value)
"""
# POSIX‐only imports; will be skipped on Windows
import re
try:
    import resource
    import os
except ImportError:
    resource = None
    os = None


def _parse_mem_limit(limit):
    """
    Parse a k8s-style memory string into bytes.
    Accepts:
      - int or float (interpreted as bytes)
      - str: "<number><suffix>", where suffix is one of:
          K, KB, Ki, KiB, M, MB, Mi, MiB, G, GB, Gi, GiB, ...
    Allows decimal or binary prefixes and optional 'B'.
    """
    if isinstance(limit, (int, float)):
        return int(limit)
    s = limit.upper().strip()
    m = re.fullmatch(r"(\d+(\.\d+)?)([KMGTPEmgtpe][iI]?[bB]?)?", s)
    if not m:
        raise ValueError(f"Invalid memory limit format: {limit!r}")
    number = float(m.group(1))
    unit = (m.group(3) or "").upper()

    # mapping suffix → multiplier
    mult = {
        "K":   10**3, "KB":  10**3,
        "KI":  2**10, "KIB": 2**10,
        "M":   10**6, "MB":  10**6,
        "MI":  2**20, "MIB": 2**20,
        "G":   10**9, "GB":  10**9,
        "GI":  2**30, "GIB": 2**30,
        "T":   10**12, "TB": 10**12,
        "TI":  2**40, "TIB": 2**40,
        "P":   10**15, "PB": 10**15,
        "PI":  2**50, "PIB": 2**50,
        "E":   10**18, "EB": 10**18,
        "EI":  2**60, "EIB": 2**60,
        "":    1,
    }
    if unit not in mult:
        raise ValueError(f"Unknown memory unit: {unit!r}")
    return int(number * mult[unit])


def _parse_cpu_limit(limit):
    """
    Parse a k8s-style CPU string into a float number of CPUs.
      - "250m" ⇒ 0.25
      - "1"    ⇒ 1.0
      - 2      ⇒ 2.0
    """
    if isinstance(limit, (int, float)):
        return float(limit)
    s = limit.strip()
    m = re.fullmatch(r"(\d+(\.\d+)?)(m?)", s, re.IGNORECASE)
    if not m:
        raise ValueError(f"Invalid CPU limit: {limit!r}")
    number = float(m.group(1))
    if m.group(3).lower() == "m":
        return number / 1000.0
    return number


def _limit_resources_posix(mem_limit=None, cpu_time=None, cpu_cores=None, nice=0):
    """
    Called in the child (via preexec_fn) on Linux/macOS.
    - mem_limit: int bytes or k8s-style str (e.g. "1Gi", "512M", "8G")
    - cpu_time: int seconds; maximum CPU-seconds before SIGXCPU
    - cpu_cores: iterable of core indices (Linux only)
    - nice: added niceness
    """
    # 1) Lower priority
    if nice and os:
        os.nice(nice)

    # 2) Memory limit
    if mem_limit is not None and resource:
        try:
            bytes_limit = _parse_mem_limit(mem_limit)
            resource.setrlimit(resource.RLIMIT_AS, (bytes_limit, bytes_limit))
        except Exception:
            # Some platforms (e.g. macOS) may ignore or reject RLIMIT_AS
            pass

    # 3) CPU‐seconds limit
    if cpu_time and resource:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_time, cpu_time))

    # 4) CPU‐core affinity (Linux only)
    if cpu_cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(cpu_cores))


def setup_local_process(mem_limit=None, cpu_limit=None, cpu_time=None, nice=0):
    """
    Set up the subporcess pre-exec function to set resource limits, ...etc.

    Arguments:
      mem_limit – int/float bytes or k8s-style string ("8Gi", "512Mi", etc.)
      cpu       – k8s-style CPU string or number ("750m", "2", etc.)
                   *fractional CPUs are parsed but ignored; only whole cores used*
      cpu_time  – int seconds; maximum CPU-seconds before SIGXCPU
      nice      – int; niceness increment

    Returns:
      Callable preexec_fn function that gets executed after the fork() and before exec()
    """
    preexec = None
    # --- Parse memory ---
    mem_bytes = None
    if mem_limit is not None:
        mem_bytes = _parse_mem_limit(mem_limit)

    # --- Parse CPU and select cores ---
    cpu_cores = None
    if cpu_limit is not None:
        cpu_units = _parse_cpu_limit(cpu_limit)
        total_cores = os.cpu_count() or 1
        # Only the integer part is used for affinity
        n_cores = min(int(cpu_units), total_cores)
        cpu_cores = list(range(n_cores)) if n_cores > 0 else []

    # --- Build preexec_fn for POSIX ---
    preexec = None
    if mem_bytes is not None or cpu_cores is not None or nice:
        def preexec_fn():
            return _limit_resources_posix(
            mem_limit=mem_bytes,
            cpu_time=cpu_time,
            cpu_cores=cpu_cores,
            nice=nice,
        )
        preexec = preexec_fn

    return preexec
