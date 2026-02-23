---
title: livetrainer-hot-reloading
authors:
  - "@szaher"
reviewers:
  - TBD
approvers:
  - TBD
creation-date: 2025-06-01
last-updated: 2025-07-01
status: implementable
see-also:
  - "/docs/proposals/2-trainer-local-execution"
---

# LiveTrainer: Hot-Reloading Hyperparameters During Distributed Training

## Summary

This enhancement introduces **LiveTrainer**, a new trainer type for the Kubeflow Trainer v2 SDK that enables real-time modification of optimizer hyperparameters (learning rate, momentum, weight decay, etc.) during active distributed training jobs — without stopping, checkpointing, or restarting the training process. LiveTrainer monitors a YAML configuration file on a shared ReadWriteMany (RWX) volume. When rank 0 detects a file change, it broadcasts the updated parameters to all distributed workers via `torch.distributed.broadcast`, and each worker applies the updates directly to its optimizer param groups. The user edits the YAML file from a notebook or CLI, and the training job picks up the changes at the next configurable sync interval.

## Motivation

### Problem Statement

Long-running distributed training jobs (hours to days) frequently require hyperparameter adjustments mid-run. The current options are all costly:

1. **Stop, edit, restart**: Wastes GPU hours, requires checkpoint/resume logic, risks losing optimizer state (momentum buffers, Adam moving averages), and resets data loader positions. A 3-day pre-training run may lose 6+ hours of warmup if restarted.
2. **Learning rate schedulers**: Pre-defined schedules (cosine decay, step decay) cannot react to emergent training dynamics. A loss plateau at step 50K may need a manual LR reduction that no pre-programmed schedule anticipated.
3. **External hyperparameter services**: Solutions like Ray Tune or Optuna add significant infrastructure complexity (dedicated servers, databases, agent sidecars) and are designed for automated search, not human-in-the-loop adjustments.

ML practitioners frequently need to "watch the loss curve and tweak the learning rate" without the overhead of a full hyperparameter optimization framework or a job restart.

### Goals

1. Allow users to modify optimizer hyperparameters (learning rate, momentum, weight decay, etc.) while a distributed training job is running on Kubernetes.
2. Provide a simple, file-based interface: edit a YAML file on a shared volume, and training picks up the change at the next sync interval.
3. Guarantee consistency across all distributed training ranks via atomic broadcast from rank 0.
4. Require zero additional infrastructure beyond a shared filesystem, which is already common in HPC and Kubernetes ML clusters.
5. Keep the training pod runtime dependency-free from the Kubeflow SDK — the injected instrumentation code must be self-contained.
6. Integrate as a first-class trainer type within the existing `TrainerClient.train()` API, alongside `CustomTrainer` and `BuiltinTrainer`.

### Non-Goals

1. **Automated hyperparameter search.** LiveTrainer is human-in-the-loop. Automated search (Bayesian optimization, population-based training) should use Kubeflow Optimizer or Ray Tune.
2. **Model architecture changes at runtime.** Only optimizer param group scalar values are modified. Model structure, loss function, and data pipeline are fixed at job start.
3. **Gradient or weight manipulation.** LiveTrainer modifies optimizer configuration, not model weights or gradients directly.
4. **Non-PyTorch frameworks.** The initial implementation targets PyTorch optimizers. JAX, TensorFlow, and other frameworks are deferred.
5. **Volume provisioning.** LiveTrainer does not create, mount, or manage shared volumes. The user or platform provisions the RWX volume independently.

## Proposal

### User Stories

#### Story 1: Learning Rate Tuning During Pre-training

> As an ML engineer pre-training a transformer model on a multi-GPU cluster, I want to lower the learning rate when I observe a loss plateau in my monitoring dashboard, so that I can resume convergence without killing the 3-day training job and losing 6 hours of warmup.

#### Story 2: Momentum Adjustment During Fine-tuning

> As a researcher fine-tuning a vision model, I want to increase momentum from 0.9 to 0.95 after the first 10 epochs because the gradient noise has stabilized, so that I can accelerate convergence without restarting and re-loading the model checkpoint.

#### Story 3: Emergency Weight Decay Correction

> As an ML ops engineer, I notice that training loss is diverging due to an incorrectly configured weight decay. I want to correct it immediately, so that I can save the run without losing the current optimizer state (Adam moving averages accumulated over 50K steps).

#### Story 4: Collaborative Tuning from a Notebook

> As a data scientist using a Kubeflow Notebook, I want to edit a `params.yaml` file on a shared volume and have my running training job pick up the changes automatically, so that I can iterate on hyperparameters interactively without leaving my notebook environment.

### User-Facing API

```python
from kubeflow.trainer import TrainerClient, LiveTrainer, SyncConfig

def train_fn():
    import torch
    import torch.nn as nn

    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        for step in range(1000):
            # Injected globals: _kubeflow_sync_loop, _apply_param_updates
            if _kubeflow_sync_loop.should_sync():
                updates = _kubeflow_sync_loop.sync_params()
                if updates:
                    _apply_param_updates(optimizer, updates)

            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

trainer = LiveTrainer(
    func=train_fn,
    shared_volume_mount_path="/mnt/shared",
    hot_reload_params=["learning_rate", "momentum", "weight_decay"],
    sync_config=SyncConfig(
        sync_interval=10,
        cooldown_seconds=0.5,
        checksum_validation=True,
    ),
    num_nodes=4,
    resources_per_node={"gpu": 2},
)

client = TrainerClient()
client.train(trainer=trainer)
```

To update parameters while training is running:

```yaml
# /mnt/shared/params.yaml
learning_rate: 0.001
momentum: 0.95
```

### Workflow Description

```
                writes params.yaml
   User / Notebook ──────────────────► Shared Volume (/mnt/shared/params.yaml)
                                            │
                                            │ FileWatcher polls mtime
                                            │ every sync_interval steps
                                            ▼
                                      Training Pod (Rank 0)
                                            │
                                            │ torch.distributed.broadcast()
                                            │ (length tensor + payload tensor)
                                            ▼
                                      Training Pods (Rank 1..N)
                                            │
                                            │ apply_param_updates(optimizer, updates)
                                            ▼
                                      Optimizer param groups updated
```

### API Extensions

LiveTrainer is registered as a new trainer type in `TrainerClient.train()`:

```python
def train(
    self,
    trainer: CustomTrainer
        | CustomTrainerContainer
        | BuiltinTrainer
        | LiveTrainer        # ← new
        | None = None,
    ...
) -> str:
```

The Kubernetes backend dispatches LiveTrainer alongside existing trainer types:

```python
elif isinstance(trainer, LiveTrainer):
    if runtime.trainer.trainer_type != TrainerType.CUSTOM_TRAINER:
        raise ValueError(f"LiveTrainer can't be used with {runtime} runtime")
    trainer_cr = live_trainer_utils.get_trainer_cr_from_live_trainer(runtime, trainer)
```

No new CRDs, API groups, or operator changes are required. LiveTrainer generates a standard `TrainerV1alpha1Trainer` CRD with an instrumented command — it is purely a client-side SDK feature.

### Implementation Details/Notes/Constraints

#### Architecture: Two-Phase Design

The implementation separates into **build time** (SDK, client-side) and **run time** (training pod):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SDK (Build Time)                                 │
│                                                                            │
│  types.py ──► utils.py ──► runtime.py ──► Trainer CRD                    │
│  (config)     (CRD gen)    (code gen)     (Kubernetes resource)           │
│                                                                            │
│  1. inspect.getsource(train_fn)                                            │
│     → extract user training function as Python source                      │
│  2. inspect.getsource(_create_live_trainer_instrumentation)               │
│     → extract FileWatcher + SyncControlLoop + apply_param_updates          │
│  3. Concatenate instrumentation wrapper + user code                        │
│     → embed as the CRD command string                                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ TrainJob CR submitted to K8s
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Training Pod (Run Time)                             │
│                                                                            │
│  Self-contained Python script (no Kubeflow SDK dependency):                │
│                                                                            │
│  ┌──────────────┐  polls   ┌──────────────────┐                           │
│  │  FileWatcher  │◄────────│ SyncControlLoop  │◄── user calls             │
│  │  (mtime+md5)  │         │ (step counter)   │    should_sync()          │
│  └──────┬───────┘         └────────┬─────────┘    sync_params()           │
│         │ params dict              │ broadcast                             │
│         ▼                          ▼                                       │
│  apply_param_updates()    torch.distributed.broadcast()                    │
│  (optimizer update)       (rank 0 → all workers)                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Component Breakdown

**`types.py` — Configuration Dataclasses**

| Dataclass | Purpose |
|-----------|---------|
| `LiveTrainer` | Main trainer config: training function, shared volume path, parameter whitelist, sync config, num_nodes, resources, image, env |
| `SyncConfig` | Sync loop behavior: step interval, cooldown delay, config/lock file names, checksum validation |

Both validate inputs in `__post_init__` and raise `ValueError` with descriptive messages.

**`runtime.py` — Code Generation via Source Extraction**

- `_create_live_trainer_instrumentation()` defines `FileWatcher`, `SyncControlLoop`, and `apply_param_updates` as nested classes/functions inside a single outer function.
- `get_live_trainer_instrumentation_wrapper()` extracts the outer function as source via `inspect.getsource()`, appends a parameterized call, and produces a complete Python script with a `{{user_func_import_and_call}}` placeholder.

**`utils.py` — CRD Generation**

`get_trainer_cr_from_live_trainer()` builds the `TrainerV1alpha1Trainer` CRD by extracting the user function source, generating the wrapper, assembling the command (handling `torchrun` templates and pip package installation), and setting CRD fields.

**`constants.py` — Default Values**

| Constant | Value |
|----------|-------|
| `DEFAULT_SHARED_VOLUME_MOUNT_PATH` | `"/mnt/shared"` |
| `DEFAULT_CONFIG_FILE_NAME` | `"params.yaml"` |
| `DEFAULT_LOCK_FILE_NAME` | `"params.yaml.lock"` |

#### Distributed Broadcast Protocol

In multi-node training, only rank 0 reads the filesystem. Updates propagate via a two-step `torch.distributed.broadcast`:

1. **Length broadcast**: Rank 0 serializes filtered params to JSON bytes, broadcasts the byte count as a `torch.long` tensor. Length 0 means no update — all ranks return early.
2. **Payload broadcast**: Rank 0 broadcasts the JSON bytes as a `torch.uint8` tensor. All ranks decode identically.

In single-process mode (`torch.distributed` not initialized), the sync loop reads directly from FileWatcher without broadcast.

#### File Change Detection Layers

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| 1. mtime check | `os.path.getmtime()` | Fast reject — skip if file not touched |
| 2. Cooldown | `time.sleep(cooldown_seconds)` | Wait after mtime change for writer to finish |
| 3. MD5 checksum | `hashlib.md5()` | Skip if content unchanged (handles mtime-only touches) |
| 4. Lock file (optional) | Check `params.yaml.lock` existence | Writer creates lock before write, removes when done |

#### Parameter Application Mapping

| YAML Key | Optimizer Param Group Key |
|----------|--------------------------|
| `learning_rate` | `lr` (mapped) |
| `momentum` | `momentum` (direct) |
| `weight_decay` | `weight_decay` (direct) |
| Any other key | Applied directly if found in param groups |

Unknown keys produce a log warning but do not raise exceptions, ensuring training stability.

## Design Decisions

### Decision 1: Polling vs. inotify/fsnotify

**Chosen: Polling via `os.path.getmtime()`**

| Criterion | Polling (mtime) | inotify/fsnotify |
|-----------|-----------------|------------------|
| NFS compatibility | Works universally | **Does not work** — inotify is local to the kernel |
| CephFS / Lustre / GlusterFS | Works universally | Unreliable or unsupported |
| Local filesystem | Works | Works |
| Detection latency | `sync_interval * step_time` | Near-instant |
| CPU overhead | Negligible (single `stat()` syscall) | None (kernel-driven) |
| Implementation complexity | Simple | Requires platform-specific libraries (`pyinotify`, `watchdog`) |
| External dependencies | None | `pyinotify` or `watchdog` package |

**Pros:**
- Universal filesystem compatibility — works on NFS, CephFS, Lustre, GlusterFS, and local filesystems
- Zero external dependencies
- Simple implementation, easy to reason about correctness
- `stat()` overhead is negligible vs. training step cost (~1ms NFS, <0.1ms local)

**Cons:**
- Detection latency bounded by `sync_interval * step_time`
- Wastes a `stat()` syscall on every sync interval even when no update exists

**Risks:**
- NFS attribute caching (`acregmax`, default 60s) can delay mtime visibility across clients

**Mitigation:**
- Users can tune NFS mount options (`noac`, `acregmax=0`) for lower latency
- MD5 checksum layer provides defense-in-depth against stale mtime
- Human-in-the-loop latency tolerance is seconds to minutes, well within bounds

### Decision 2: Source Code Injection vs. SDK Runtime Import

**Chosen: `inspect.getsource()` extraction and injection**

| Criterion | Source Injection | SDK Import at Runtime |
|-----------|-----------------|----------------------|
| Runtime dependencies | stdlib + PyYAML only | Full Kubeflow SDK (~50MB+) |
| Container image size | No SDK overhead | SDK must be in image |
| Version coupling | None — injected code is frozen at build time | SDK version in image must match |
| Debuggability | Full source visible in pod (can `cat` the script) | Import chain opaque |
| Testability | Function directly testable in SDK unit tests | Same |
| IDE support | Full syntax highlighting, type checking in SDK | Same |
| Fragility | Breaks on C extensions, lambdas, REPL-defined functions | Stable imports |

**Pros:**
- Training pods have zero Kubeflow SDK dependency — only stdlib + PyYAML + torch
- No version coupling between SDK and training container
- Injected code is fully visible and debuggable in the pod
- Keeps container images lean

**Cons:**
- `inspect.getsource()` cannot extract lambdas, C extensions, or dynamically generated functions
- Generated script can be large if the instrumentation function grows
- Refactoring the instrumentation function changes the injected code — must ensure backward compatibility

**Risks:**
- Users defining training functions in a REPL or via `exec()` will get `OSError: could not get source code`

**Mitigation:**
- `LiveTrainer.__post_init__` validates `func` is callable
- Failure occurs at CRD generation time (client-side), not at pod runtime — fast feedback loop
- Documented requirement: function must be defined in a `.py` file or Jupyter notebook cell

### Decision 3: Storage-Agnostic Design vs. Built-in NFS Provisioning

**Chosen: Storage-agnostic (user provisions RWX volume externally)**

| Criterion | Storage-Agnostic | Built-in NFS Provisioning |
|-----------|-----------------|---------------------------|
| User setup effort | Must provision RWX volume separately | Provide NFS server IP + path |
| Storage flexibility | Any RWX backend (NFS, CephFS, Lustre, EFS, etc.) | NFS only |
| SDK complexity | Minimal — single `shared_volume_mount_path` field | NFS-specific fields, volume override logic |
| Platform integration | Works with existing PVC/StorageClass workflows | Bypasses platform storage policies |
| Testability | Easier to mock (any temp directory) | Requires NFS-specific mocking |
| Separation of concerns | Trainer focuses on training, storage is external | Trainer manages storage lifecycle |

**Pros:**
- Works with any RWX storage technology, present and future
- Simpler SDK code — no `build_nfs_volume_overrides()`, no `nfs_server`/`nfs_path` fields
- Respects platform storage policies (admins control which StorageClass is used)
- Cleaner separation of concerns

**Cons:**
- Higher initial setup burden on users — must provision the volume themselves
- First-time users may not know how to attach an RWX PVC to a TrainJob

**Risks:**
- Users unfamiliar with Kubernetes PVCs may struggle with volume setup

**Mitigation:**
- Example notebook documents `PodTemplateOverrides` pattern for attaching PVCs
- README covers common patterns: NFS PVC, CephFS, platform-managed volumes
- `ClusterTrainingRuntime` can pre-configure the volume in cluster-level templates

### Decision 4: Step-Based Sync Interval vs. Wall-Clock Timer

**Chosen: Step-based interval (check every N training steps)**

| Criterion | Step-Based | Wall-Clock Timer |
|-----------|-----------|-----------------|
| Determinism | Deterministic, reproducible | Non-deterministic |
| Implementation | Integer increment + modulo | Background thread + locks |
| Thread safety | Single-threaded, no locks | Requires mutex on optimizer |
| Overhead | Negligible (1 integer op per step) | Thread management overhead |
| User mental model | "Check every 10 steps" | "Check every 5 seconds" |
| Slow-step behavior | Latency = `sync_interval * step_time` | Constant latency |

**Pros:**
- Zero threading complexity — runs inline in the training loop
- Deterministic behavior makes debugging and testing straightforward
- No risk of concurrent optimizer access
- Negligible per-step overhead

**Cons:**
- Detection latency scales with step duration (slow steps = slow detection)
- Not suitable for variable-duration steps (some steps much longer than others)

**Risks:**
- With slow steps (2 minutes each) and `sync_interval=10`, worst-case latency is 20 minutes

**Mitigation:**
- Users with slow steps set `sync_interval=1` — cost is one `stat()` per step, negligible vs. step time
- Documentation clearly explains the latency tradeoff

### Decision 5: Rank 0 Read + Broadcast vs. All Ranks Read Independently

**Chosen: Rank 0 reads, broadcasts to all workers**

| Criterion | Rank 0 + Broadcast | All Ranks Read |
|-----------|-------------------|----------------|
| Filesystem load | 1 `stat()` + 1 `read()` per interval | N `stat()` + N `read()` per interval |
| Consistency | Guaranteed identical across all ranks | Possible race conditions (stale caches) |
| NFS cache effects | Single client, predictable | Multiple clients, different cache states |
| Failure mode | If rank 0 dies, no updates (but job already failed) | Partial updates possible |
| Dependencies | Requires `torch.distributed` for multi-node | No `torch.distributed` needed |

**Pros:**
- Atomic consistency — all ranks receive identical updates at the same logical point
- Minimizes filesystem load (1 reader vs. N readers)
- Avoids NFS attribute cache inconsistencies across clients
- Aligns with PyTorch DDP convention where rank 0 is the coordinator

**Cons:**
- Single point of dependency on rank 0 for update detection
- Adds `torch.distributed.broadcast` calls (2 per sync: length + payload)

**Risks:**
- Broadcast adds ~1ms overhead per sync interval — negligible vs. gradient all-reduce

**Mitigation:**
- Fallback to direct read in single-process mode (no broadcast needed)

### Decision 6: YAML File on Shared Volume vs. gRPC/REST API

**Chosen: YAML file on shared volume**

| Criterion | YAML File | gRPC/REST API |
|-----------|-----------|---------------|
| Infrastructure required | Shared filesystem (already exists) | Dedicated server, service discovery, ports |
| User tooling | Any text editor, `echo`, notebook | Client library, curl, authentication |
| Atomicity | Lock file protocol (cooperative) | Native RPC semantics |
| Offline persistence | File persists across pod restarts | Server must be running |
| Debugging | `cat /mnt/shared/params.yaml` | API call or log inspection |
| Security model | Filesystem permissions (POSIX) | Auth tokens, TLS certificates |
| Scalability | Single file, one reader | Server can handle many clients |

**Pros:**
- Zero additional infrastructure — shared filesystem already exists in most ML clusters
- Universal tooling — edit from notebook, shell, kubectl exec, any language
- File persists independently of training pod lifecycle
- Simple to debug and inspect

**Cons:**
- No delivery guarantee — no acknowledgment that the training job received the update
- File-based communication is unstructured compared to typed RPC
- Lock file protocol is cooperative (no enforcement if writer doesn't use it)

**Risks:**
- Corrupted YAML (syntax errors, partial writes) could prevent updates

**Mitigation:**
- Three-layer integrity protection (cooldown, MD5 checksum, lock file)
- Invalid YAML is logged and skipped — training continues with previous params
- Pod logs confirm when updates are applied

### Decision 7: Injected Globals vs. Function Parameter Passing

**Chosen: Inject `_kubeflow_sync_loop` and `_apply_param_updates` as module-level globals**

| Criterion | Injected Globals | Function Parameters |
|-----------|-----------------|-------------------|
| User function signature | Unchanged | Must add new parameters |
| Backward compatibility | Same function works with `CustomTrainer` | Different signature needed |
| Discoverability | Less discoverable (implicit globals) | Explicit in function signature |
| IDE autocompletion | No autocompletion for injected names | Full autocompletion |
| Linter warnings | Requires `# noqa: F821` | No warnings |
| Testability | Globals must be set up before calling | Pass mocks as arguments |

**Pros:**
- User's training function signature is unchanged — same function works with `CustomTrainer` and `LiveTrainer`
- No changes to the `func` field contract in `LiveTrainer`
- Wrapper handles initialization before user code executes

**Cons:**
- Magic globals reduce code clarity
- IDE autocompletion does not work for `_kubeflow_sync_loop` and `_apply_param_updates`
- Linter reports `F821 undefined name` without `# noqa` annotations

**Risks:**
- Users may accidentally shadow the globals with local variables of the same name

**Mitigation:**
- Underscore-prefixed names (`_kubeflow_sync_loop`) signal "framework-managed, do not override"
- Documentation and example notebook show the exact integration pattern
- `_create_live_trainer_instrumentation()` is directly callable in tests for local validation

## Risks and Mitigations

### Critical

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| C1 | **Silent parameter drift across ranks** — ranks apply updates at different training steps, causing divergent model weights | Training produces incorrect or non-reproducible results | Low | Broadcast protocol guarantees all ranks receive identical updates atomically. `should_sync()` is called at the same training step on all ranks because DDP synchronizes via gradient all-reduce. |
| C2 | **Training destabilization** — user sets a dangerously invalid hyperparameter value (e.g., `lr=100`, `weight_decay=-1`) | Loss divergence, NaN gradients, wasted GPU hours | Medium | `hot_reload_params` whitelist limits *which* parameters can be changed. Value range validation (min/max) is deferred to beta. The user is trusted in a human-in-the-loop workflow, but guardrails are planned. |
| C3 | **Source extraction failure at CRD generation** — `inspect.getsource()` cannot extract the user's training function | `TrainerClient.train()` raises an exception, job is never submitted | Low | Fails at client-side CRD generation, not at pod runtime — fast feedback. Documented requirement: function must be in a `.py` file or notebook cell. Clear error message guides the user. |

### Moderate

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| M1 | **Stale NFS attribute cache** — mtime not visible across NFS clients for up to 60 seconds (default `acregmax`) | Parameter update detection delayed by up to 60s beyond the sync interval | Medium | Cooldown + MD5 checksum layers provide defense-in-depth. Users can tune NFS mount options (`noac`, `acregmax=0`). Human-in-the-loop latency tolerance is typically seconds to minutes. |
| M2 | **Partial YAML read** — file read while writer is mid-write | YAML parse error, update skipped for this sync cycle | Medium | Three-layer protection: cooldown delay (default 0.5s), MD5 checksum (detects content change), lock file protocol (cooperative atomic writes). YAML parse errors are logged and skipped — training continues. |
| M3 | **PyYAML not available in training container** — base image lacks `pyyaml` package | Instrumentation crashes at pod startup with `ModuleNotFoundError` | Low | Most PyTorch images include PyYAML. If missing, user adds `pyyaml` to `packages_to_install`. |
| M4 | **User modifies function source after LiveTrainer creation but before `train()` call** — `inspect.getsource()` reads stale source | Injected code does not match user's intent | Low | Standard Python `inspect` behavior — reads from source file at extraction time. In Jupyter, cell source is current when `getsource()` runs. |

### Low

| # | Risk | Impact | Likelihood | Mitigation |
|---|------|--------|------------|------------|
| L1 | **Polling overhead on high-frequency sync** — `stat()` call on every step with `sync_interval=1` | Negligible performance impact (~1ms per call) | N/A | A `stat()` syscall takes ~1ms on NFS, <0.1ms local. Negligible vs. training step cost (10ms–10s). |
| L2 | **Large YAML config files** — user places extensive config beyond hyperparameters | Unnecessary memory usage, slower parse time | Low | File is read entirely into memory. For reasonable hyperparameter configs (<1KB), no issue. |
| L3 | **Concurrent writers** — multiple users edit `params.yaml` simultaneously | Last writer wins, potentially confusing | Low | Lock file protocol mitigates. Human-in-the-loop tool — concurrent writes are unlikely in practice. |

## Alternatives

### Alternative 1: Kubernetes ConfigMap-Based Parameter Updates

Mount a ConfigMap as a volume; update the ConfigMap via `kubectl edit configmap`.

| Criterion | ConfigMap | Shared Volume (Chosen) |
|-----------|-----------|----------------------|
| Infrastructure | Kubernetes API server | Shared filesystem |
| Propagation delay | ~60s (kubelet sync frequency) | `sync_interval * step_time` |
| Size limit | 1MB per ConfigMap | Filesystem limits |
| Update mechanism | `kubectl edit`/`kubectl patch` | Any file editor |
| RBAC | Kubernetes RBAC | POSIX filesystem permissions |

**Rejected because:** ConfigMap propagation delay is controlled by kubelet `--sync-frequency` (default 60s) and is not configurable per-pod. This introduces unpredictable, high latency compared to direct file access. ConfigMap updates also add API server load proportional to the number of pods watching the ConfigMap.

### Alternative 2: Sidecar Parameter Server (gRPC/HTTP)

Run a sidecar container in each training pod that exposes a gRPC or HTTP API. A central controller pushes parameter updates to all sidecars.

**Rejected because:** Adds a sidecar container, port allocation, service discovery, and health monitoring to every training pod. The sidecar needs its own resource allocation and failure handling. This is significantly more complex than reading a file and offers marginal benefit for a human-in-the-loop workflow where updates happen at most a few times per hour.

### Alternative 3: Redis / etcd Pub/Sub

Use a shared key-value store (Redis, etcd) with pub/sub notifications for parameter updates.

**Rejected because:** Requires deploying and maintaining additional infrastructure (Redis server or etcd cluster). Introduces a new failure domain — if Redis is unavailable, no updates can be delivered. Adds network round-trip latency. The file-based approach requires zero additional infrastructure beyond what already exists in most ML clusters.

### Alternative 4: Shared Memory / POSIX Signals

Use POSIX shared memory segments (`shm`) or Unix signals (`SIGUSR1`) to communicate parameter updates between processes.

**Rejected because:** Neither mechanism works across network boundaries in distributed training. Shared memory is node-local. POSIX signals carry no payload (only a signal number). Both are restricted to single-machine scenarios and cannot support multi-node distributed training.

### Alternative 5: Built-in NFS Volume Provisioning (Original Design)

The original implementation included `nfs_server` and `nfs_path` fields on `LiveTrainer` and a `build_nfs_volume_overrides()` function that generated Kubernetes volume/volumeMount patches for the pod template.

**Rejected because:** Coupled the SDK to NFS as the only supported storage backend. Duplicated volume management logic that already exists in Kubernetes (PVCs, StorageClasses, `PodTemplateOverrides`). Prevented use of CephFS, Lustre, EFS, or platform-managed storage. The storage-agnostic design using `shared_volume_mount_path` is simpler and more flexible.

## Test Plan

### Unit Tests

All components have comprehensive unit tests co-located with source files:

| Test File | Scope | Key Test Cases |
|-----------|-------|----------------|
| `types_test.py` | `SyncConfig` and `LiveTrainer` validation | Defaults, custom values, all error paths (`ValueError` for invalid sync_interval, cooldown, func, mount path, hot_reload_params, sync_config type) |
| `utils_test.py` | CRD generation | Command building, instrumentation markers in output, num_nodes, resources, env, image, func_args passthrough, wrapper code compilation |
| `runtime_test.py` | FileWatcher, SyncControlLoop, apply_param_updates, wrapper generation | mtime detection, lock file blocking, MD5 checksum dedup, invalid YAML handling, missing file handling, step interval counting, param whitelist filtering, empty whitelist allows all, `learning_rate`→`lr` mapping, momentum/weight_decay application, unknown param warning, wrapper compilation, config path embedding |

Tests requiring PyTorch are conditionally skipped via `@pytest.mark.skipif`.

```bash
# All LiveTrainer tests
uv run pytest kubeflow/trainer/livetrainer/ -v

# With coverage
uv run coverage run -m pytest kubeflow/trainer/livetrainer/ && uv run coverage report
```

### Integration Tests (Planned for Beta)

- Submit a LiveTrainer job to a kind cluster with an NFS-backed RWX PVC
- Write a `params.yaml` update from a test harness pod
- Assert training pod logs contain the parameter update confirmation
- Assert the optimizer state reflects the new parameter values via a checkpoint inspection

### Manual Validation

The `examples/livetrainer/live_trainer_demo.ipynb` notebook provides an interactive walkthrough with inline `assert` statements that serve as executable validation.

## Graduation Criteria

### Dev Preview (Alpha) — Current

- [x] Core implementation: FileWatcher, SyncControlLoop, apply_param_updates
- [x] Source code injection via `inspect.getsource()`
- [x] Distributed broadcast protocol (rank 0 → all workers)
- [x] Integration with `TrainerClient.train()` and Kubernetes backend
- [x] Comprehensive unit tests (types, utils, runtime)
- [x] Example notebook with inline assertions
- [x] Storage-agnostic design (NFS-specific logic removed)
- [x] Documentation: package README, example README, RFE proposal

### Tech Preview (Beta)

- [ ] Integration tests on a live Kubernetes cluster (kind + NFS PVC)
- [ ] Parameter value range validation (optional min/max constraints per param)
- [ ] Callback hooks for pre/post parameter update (e.g., log to W&B, trigger LR scheduler reset)
- [ ] Acknowledgment mechanism (training pod writes a receipt file confirming update applied)
- [ ] User-facing documentation on kubeflow.org

### GA (Stable)

- [ ] E2E tests in CI with NFS and CephFS storage backends
- [ ] Performance benchmarks: polling overhead at various sync intervals and cluster sizes
- [ ] Support for non-PyTorch optimizers (JAX, TensorFlow)
- [ ] Support for scheduler-aware updates (override base LR, let scheduler compute effective rate)
- [ ] Multi-optimizer targeting (by name, for GANs and multi-task models)

## Upgrade / Downgrade Strategy

LiveTrainer is a purely client-side SDK feature. No operator, CRD, or cluster-side changes are required.

- **Upgrade:** Users adopt `LiveTrainer` by updating the Kubeflow SDK version. Existing `CustomTrainer` and `BuiltinTrainer` workflows are unaffected.
- **Downgrade:** Users revert to a previous SDK version. Any in-flight LiveTrainer jobs continue running (the injected code is self-contained in the pod), but new LiveTrainer jobs cannot be submitted.
- **No migration required:** LiveTrainer adds new types; it does not modify existing types or APIs.

## Version Skew Strategy

Not applicable. LiveTrainer does not interact with the Kubeflow Trainer operator or any server-side components beyond submitting a standard `TrainJob` CR. The injected instrumentation code runs entirely within the training pod and has no version dependency on the SDK or operator.

## Operational Aspects of API Extensions

### Failure Modes

| Failure | Detection | Impact | Recovery |
|---------|-----------|--------|----------|
| Shared volume not mounted | Pod startup crash (path does not exist) | Job fails to start | User provisions the RWX volume and re-submits |
| Config file has invalid YAML | Warning log in training pod | Update skipped, training continues with previous params | User fixes the YAML syntax |
| Config file not created yet | `sync_params()` returns `None` | No-op, training runs with initial params | User creates the config file when ready |
| Shared filesystem unavailable mid-training | `OSError` caught, warning logged | Updates paused until filesystem recovers | Filesystem recovery restores normal operation |
| `torch.distributed` not initialized | Detected at SyncControlLoop init | Falls back to single-process mode (direct read) | No action needed — correct behavior |

### Support Procedures

**Q: How do I verify that my parameter update was applied?**

Check the training pod logs. Successful updates produce:
```
[LiveTrainer] Updated lr = 0.001
```

**Q: Why is my update not being picked up?**

1. Verify the YAML file is on the correct path: `kubectl exec <pod> -- cat /mnt/shared/params.yaml`
2. Check `sync_interval` — updates are only detected every N steps
3. Check for NFS attribute caching — try `ls -la /mnt/shared/params.yaml` from the pod to force a cache refresh
4. Check for lock files: `ls /mnt/shared/params.yaml.lock` — if present, the reader skips

**Q: Can I use LiveTrainer with the local process or container backend?**

Currently, LiveTrainer is supported only with the Kubernetes backend. Local process and container backend support is deferred.

## Implementation History

| Date | Milestone |
|------|-----------|
| 2025-06-01 | Initial proposal and implementation with NFS-specific fields |
| 2025-07-01 | Refactored to storage-agnostic design — removed `nfs_server`, `nfs_path`, `build_nfs_volume_overrides()`, renamed `nfs_mount_path` → `shared_volume_mount_path` |

## Drawbacks

1. **User must provision shared storage.** Unlike a fully managed solution, the user or platform admin must set up the RWX volume. This is a deliberate tradeoff for storage-backend flexibility.
2. **No delivery acknowledgment.** The user has no programmatic way to confirm the training job received the update, short of watching pod logs. A receipt file mechanism is planned for beta.
3. **PyTorch-only.** The `apply_param_updates` function and broadcast protocol are PyTorch-specific. JAX and TensorFlow support requires additional implementation.
4. **Magic globals.** The `_kubeflow_sync_loop` and `_apply_param_updates` injected globals are not discoverable via IDE autocompletion and require documentation for new users.
5. **File-based communication is inherently unstructured.** There is no schema validation on the YAML beyond "must be a mapping." Users can write any key-value pair; only `hot_reload_params` filtering prevents unintended params from reaching the optimizer.

## Infrastructure Needed

No new infrastructure is required beyond what already exists in a standard Kubeflow Trainer v2 deployment:

- Kubeflow Trainer v2 operator (existing)
- A `ClusterTrainingRuntime` with PyTorch support (existing)
- A ReadWriteMany (RWX) volume accessible from training pods and the user's notebook (user-provisioned)

No new CRDs, operators, controllers, or services are introduced.
