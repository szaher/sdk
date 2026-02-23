# LiveTrainer Package

The `livetrainer` package implements hot-reloading of hyperparameters for distributed PyTorch training jobs running on Kubeflow Trainer v2. It allows users to modify optimizer parameters (learning rate, momentum, weight decay, etc.) in real time by editing a YAML file on a shared volume, without stopping or restarting training.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SDK (build time)                              │
│                                                                            │
│  types.py ──► utils.py ──► runtime.py ──► Trainer CRD with instrumented   │
│  (config)     (CRD gen)    (code gen)     command sent to Kubernetes       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ TrainJob submitted via TrainerClient
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Training Pod (run time)                            │
│                                                                            │
│  Injected instrumentation code (from runtime.py):                          │
│                                                                            │
│  ┌──────────────┐    polls mtime    ┌──────────────────┐                   │
│  │  FileWatcher  │ ◄──────────────── │  SyncControlLoop │ ◄── user code    │
│  │  (reads YAML) │                   │  (step counter)  │     calls        │
│  └──────┬───────┘                   └────────┬─────────┘     should_sync() │
│         │                                    │               sync_params() │
│         │ params dict                        │ broadcast                    │
│         ▼                                    ▼                             │
│  ┌──────────────────┐              ┌───────────────────┐                   │
│  │ apply_param_      │              │ torch.distributed │                   │
│  │ updates()         │              │ .broadcast()      │                   │
│  │ (optimizer update)│              │ (rank 0 → all)    │                   │
│  └──────────────────┘              └───────────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

The key design insight is a **two-phase architecture**:

- **Build time (SDK):** `runtime.py` uses `inspect.getsource()` to extract the `_create_live_trainer_instrumentation` function as raw Python source code. This source is concatenated with the user's training function and baked into the Trainer CRD command. This means the training pods have zero runtime dependency on the Kubeflow SDK.

- **Run time (training pod):** The injected code creates a `FileWatcher`, a `SyncControlLoop`, and an `apply_param_updates` function. These are assigned to globals (`_kubeflow_sync_loop`, `_apply_param_updates`) that the user's training function calls.

## Module Reference

### `constants.py`

Defines default values used across the package:

| Constant | Value | Purpose |
|----------|-------|---------|
| `DEFAULT_SHARED_VOLUME_MOUNT_PATH` | `"/mnt/shared"` | Default mount path for the shared RWX volume |
| `DEFAULT_CONFIG_FILE_NAME` | `"params.yaml"` | Default YAML config file name |
| `DEFAULT_LOCK_FILE_NAME` | `"params.yaml.lock"` | Default lock file name for atomic writes |

### `types.py`

Defines the two user-facing dataclasses: `LiveTrainer` and `SyncConfig`.

**`SyncConfig`** controls the synchronization behavior:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `sync_interval` | `int` | `10` | Training steps between sync checks |
| `cooldown_seconds` | `float` | `0.5` | Delay after mtime change before reading (avoids partial writes) |
| `config_file_name` | `str` | `"params.yaml"` | YAML config filename on the shared volume |
| `use_lock_file` | `bool` | `False` | Enable lock file protocol for atomic multi-key updates |
| `lock_file_name` | `str` | `"params.yaml.lock"` | Lock file name (writer creates it, deletes when done) |
| `checksum_validation` | `bool` | `True` | MD5 checksum to detect duplicate or partial content |

**`LiveTrainer`** is the main configuration object:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `func` | `Callable` | (required) | User training function |
| `shared_volume_mount_path` | `str` | `"/mnt/shared"` | Mount path for the shared RWX volume |
| `hot_reload_params` | `list[str]` or `None` | `None` | Param whitelist (`None` = allow all) |
| `sync_config` | `SyncConfig` or `None` | `None` | Sync configuration (uses defaults if `None`) |
| `func_args` | `dict` or `None` | `None` | kwargs passed to `func` |
| `image` | `str` or `None` | `None` | Container image for the TrainJob |
| `packages_to_install` | `list[str]` or `None` | `None` | Extra pip packages to install |
| `pip_index_urls` | `list[str]` | PyPI default | Pip index URLs |
| `num_nodes` | `int` or `None` | `None` | Number of distributed training nodes |
| `resources_per_node` | `dict` or `None` | `None` | Resources (e.g., `{"gpu": 2}`) |
| `env` | `dict[str, str]` or `None` | `None` | Environment variables |

Both dataclasses validate their inputs in `__post_init__` and raise `ValueError` with descriptive messages on invalid input.

### `runtime.py`

Contains two functions:

**`_create_live_trainer_instrumentation(...)`** — The self-contained instrumentation function. This function is never called directly in the SDK at build time (except in tests). Instead, its source code is extracted via `inspect.getsource()` and injected into the training pod's command. It defines three internal components:

- **`FileWatcher`** — Polls a YAML config file for changes using `os.path.getmtime()`. This polling approach is deliberate: `inotify` does not work reliably on network filesystems (NFS, CephFS, Lustre), so mtime-based polling provides universal compatibility. Supports three layers of protection against partial reads:
  1. **Cooldown delay** (`cooldown_seconds`): waits after detecting an mtime change
  2. **MD5 checksum** (`checksum_validation`): skips re-reads if content hash hasn't changed
  3. **Lock file protocol** (`use_lock_file`): skips reads while a lock file exists

- **`SyncControlLoop`** — Wraps the FileWatcher with a step counter and distributed broadcast. On each call to `should_sync()`, it increments a counter and returns `True` every `sync_interval` steps. On `sync_params()`, rank 0 reads the FileWatcher and broadcasts the serialized JSON payload to all workers via `torch.distributed.broadcast`. In single-process mode (no `torch.distributed`), it reads directly without broadcast.

- **`apply_param_updates(optimizer, updates)`** — Iterates over optimizer `param_groups` and applies matching key-value pairs. Maps the friendly name `learning_rate` to PyTorch's `lr` key. Logs warnings for unrecognized parameters but does not raise exceptions.

**`get_live_trainer_instrumentation_wrapper(trainer)`** — Takes a `LiveTrainer` instance and produces a complete Python source string. It extracts `_create_live_trainer_instrumentation` via `inspect.getsource()`, appends a call with the resolved configuration values (file paths, sync interval, etc.), and includes a `{{user_func_import_and_call}}` placeholder where the user's function code will be inserted.

### `utils.py`

**`get_trainer_cr_from_live_trainer(runtime, trainer)`** — Builds a Kubernetes Trainer CRD (`TrainerV1alpha1Trainer`) from a `LiveTrainer`. It:

1. Extracts the user's training function source via `inspect.getsource()`
2. Generates the instrumentation wrapper via `get_live_trainer_instrumentation_wrapper()`
3. Inserts the user function at the `{{user_func_import_and_call}}` placeholder
4. Builds the final command array for the runtime (e.g., `torchrun` for PyTorch)
5. Sets `num_nodes`, `resources_per_node`, `env`, and `image` on the CRD

### `__init__.py`

Exports `LiveTrainer` and `SyncConfig`. These are also re-exported from the top-level `kubeflow.trainer` package.

## Data Flow: From User Code to Running Pod

```
1. User creates LiveTrainer(func=train_fn, ...)
                    │
2. TrainerClient.train(trainer=trainer)
                    │
3. KubernetesBackend detects LiveTrainer instance
                    │
4. utils.get_trainer_cr_from_live_trainer() called
                    │
5. runtime.get_live_trainer_instrumentation_wrapper() generates wrapper code
                    │
6. inspect.getsource(train_fn) extracts user function source
                    │
7. Wrapper + user function assembled into a single Python script
                    │
8. Script embedded in Trainer CRD command: ["bash", "-c", "...python script..."]
                    │
9. TrainJob CR submitted to Kubernetes
                    │
10. Operator creates training pods, each runs the instrumented script
                    │
11. At runtime: _kubeflow_sync_loop and _apply_param_updates are available as globals
```

## Distributed Broadcast Protocol

In multi-node training, only rank 0 reads the config file. Updates are broadcast to all workers using a two-step protocol:

1. **Length broadcast:** Rank 0 serializes the filtered params dict to JSON bytes and broadcasts the byte count as a single `torch.long` tensor. If no updates, the length is 0 and all ranks return early.

2. **Payload broadcast:** Rank 0 converts the JSON bytes to a `torch.uint8` tensor and broadcasts it. All ranks decode the tensor back to bytes, parse the JSON, and return the updates dict.

This avoids requiring all ranks to access the shared filesystem and keeps the broadcast payload minimal (only changed parameters as JSON).

## Error Handling

The instrumentation handles errors gracefully to avoid crashing training:

| Scenario | Behavior |
|----------|----------|
| Config file does not exist | `sync_params()` returns `None` |
| Config file unreadable (permission error) | Logs warning, returns `None` |
| Invalid YAML | Logs warning, returns `None`, continues with previous params |
| Config is not a YAML mapping | Logs warning, returns `None` |
| Lock file present | Skips read, returns `None` |
| Unknown param key in optimizer | Logs warning, skips that key, does not crash |
| `torch.distributed` not initialized | Falls back to single-process mode (no broadcast) |

## Test Structure

Tests are co-located with the source files following the project convention:

| File | What it tests |
|------|---------------|
| `types_test.py` | `SyncConfig` and `LiveTrainer` dataclass validation — defaults, custom values, and all error paths |
| `utils_test.py` | `get_trainer_cr_from_live_trainer` — CRD generation, command building, instrumentation markers |
| `runtime_test.py` | `FileWatcher` (mtime detection, lock files, checksums, invalid YAML), `SyncControlLoop` (step intervals, param filtering), `apply_param_updates` (learning rate, momentum, weight decay mapping), and wrapper code generation/compilation |

Tests that require PyTorch (e.g., `TestApplyParamUpdates`) are marked with `@pytest.mark.skipif` and skip gracefully when torch is not installed.

Run all LiveTrainer tests:

```bash
uv run pytest kubeflow/trainer/livetrainer/ -v
```

## Design Decisions

**Why polling instead of inotify?**
Network filesystems (NFS, CephFS, Lustre, GlusterFS) do not reliably support `inotify` or `fsnotify` events. Polling via `os.path.getmtime()` works universally across all filesystem types.

**Why inject source code instead of importing the SDK?**
Training pods should not need the Kubeflow SDK installed at runtime. By extracting the instrumentation function as source code via `inspect.getsource()`, the generated script is self-contained — it only depends on `yaml`, `hashlib`, `os`, `json`, `time`, and `logging` (all stdlib except PyYAML). This also means the injected code can be inspected and debugged independently.

**Why is the trainer storage-agnostic?**
The LiveTrainer does not provision or configure volumes. The shared volume (NFS, CephFS, any RWX PVC) is mounted by the user or platform outside the SDK — via `PodTemplateOverrides`, cluster runtime templates, or platform-level configuration. This keeps the trainer focused on the training logic and avoids coupling to any specific storage technology.

**Why broadcast from rank 0 only?**
Reading the config file from every rank would create unnecessary filesystem load and could lead to inconsistencies if ranks read at slightly different times. Having rank 0 read once and broadcast ensures all workers see identical updates atomically.

**Why a step-based sync interval instead of wall-clock time?**
Step-based checking is deterministic, predictable, and doesn't require timers or threads. The cost of `should_sync()` is a single integer increment and modulo — negligible compared to a training step. Users can set `sync_interval=1` for immediate detection or a higher value to reduce filesystem polling.
