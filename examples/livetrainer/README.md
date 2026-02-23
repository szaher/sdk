# LiveTrainer Hot-Reloading Demo

This example demonstrates the **LiveTrainer** feature for Kubeflow Trainer v2 — real-time hot-reloading of hyperparameters during distributed training, without stopping or restarting the training job.

## What This Example Covers

The `live_trainer_demo.ipynb` notebook walks through:

1. **SyncConfig and LiveTrainer construction** with default and custom values
2. **Input validation** — how the dataclasses reject invalid configurations
3. **Writing a training function** that integrates with the hot-reload loop
4. **Creating a LiveTrainer** with a shared volume, parameter whitelist, and sync configuration
5. **Inspecting generated instrumentation code** — the wrapper that gets injected into training pods
6. **Local testing** of FileWatcher and SyncControlLoop without Kubernetes
7. **CRD generation** — verifying the Trainer custom resource is built correctly
8. **Submitting to Kubernetes** (commented out, for use on a live cluster)

## Prerequisites

### For running the notebook locally (sections 1-8)

- Python 3.10+
- The Kubeflow SDK installed (from this repository):
  ```bash
  cd /path/to/kubeflow-sdk
  make install-dev
  ```
- PyYAML (installed as part of the SDK dependencies)

### For submitting to Kubernetes (section 9)

- A Kubernetes cluster with the Kubeflow Trainer v2 operator installed
- `kubectl` configured to access the cluster
- A **shared RWX volume** (NFS, CephFS, or any ReadWriteMany PVC) mounted in both the notebook environment and the training pods at the same path (default: `/mnt/shared`)
- A PyTorch-compatible `ClusterTrainingRuntime` (e.g., `torch-distributed`)

## Running the Notebook

```bash
cd /path/to/kubeflow-sdk
make install-dev
jupyter notebook examples/livetrainer/live_trainer_demo.ipynb
```

Alternatively, run it with JupyterLab:

```bash
jupyter lab examples/livetrainer/live_trainer_demo.ipynb
```

All cells up to and including section 8 ("Test CRD Generation") execute locally with no cluster access. The Kubernetes submission cells in section 9 are commented out and require uncommenting plus a live cluster.

## How the Training Function Works

The training function must integrate with two injected globals: `_kubeflow_sync_loop` and `_apply_param_updates`. These are not imported — the LiveTrainer instrumentation injects them at runtime before the user function executes.

```python
def train_fn():
    import torch
    import torch.nn as nn

    model = nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.MSELoss()

    for epoch in range(100):
        for step in range(1000):
            # Check if this step should trigger a sync check
            if _kubeflow_sync_loop.should_sync():
                updates = _kubeflow_sync_loop.sync_params()
                if updates:
                    _apply_param_updates(optimizer, updates)

            # Normal training step
            x = torch.randn(32, 10)
            y = torch.randn(32, 1)
            loss = loss_fn(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

The three integration points:

| Call | Purpose |
|------|---------|
| `_kubeflow_sync_loop.should_sync()` | Returns `True` every N steps (configured by `sync_interval`). Lightweight counter check — call it on every step. |
| `_kubeflow_sync_loop.sync_params()` | Reads the config file (rank 0) and broadcasts updates to all workers. Returns a dict of updated params, or `None`. |
| `_apply_param_updates(optimizer, updates)` | Applies the param dict to optimizer param groups. Maps `learning_rate` to PyTorch's `lr` key automatically. |

## How Hot-Reload Works at Runtime

```
                writes params.yaml
   User / Notebook ──────────────────► Shared Volume (/mnt/shared)
                                            │
                                            │ polls mtime every N steps
                                            ▼
                                      Training Pod (Rank 0)
                                            │
                                            │ torch.distributed.broadcast()
                                            ▼
                                      Training Pods (Rank 1..N)
```

1. All training pods and the user's notebook mount the same RWX volume at `shared_volume_mount_path` (default: `/mnt/shared`).
2. The user writes a YAML file (default: `params.yaml`) to that volume:
   ```yaml
   learning_rate: 0.0001
   momentum: 0.99
   ```
3. Every `sync_interval` training steps, **rank 0** checks the file's `mtime`. If it changed:
   - An optional cooldown delay avoids reading partial writes.
   - An optional MD5 checksum validates content integrity.
   - An optional lock file protocol allows atomic multi-key updates.
4. Rank 0 broadcasts the update payload to all workers via `torch.distributed.broadcast`.
5. The `apply_param_updates` function maps the YAML keys to optimizer param group keys and applies them.

## LiveTrainer Configuration Reference

```python
from kubeflow.trainer import LiveTrainer, SyncConfig

trainer = LiveTrainer(
    func=train_fn,                           # Required: your training function
    shared_volume_mount_path="/mnt/shared",  # Where the RWX volume is mounted (default)
    hot_reload_params=["learning_rate", "momentum"],  # Whitelist (None = allow all)
    sync_config=SyncConfig(
        sync_interval=10,          # Check every 10 steps (default)
        cooldown_seconds=0.5,      # Wait after mtime change (default)
        config_file_name="params.yaml",     # Config filename (default)
        use_lock_file=False,                # Lock file protocol (default)
        lock_file_name="params.yaml.lock",  # Lock filename (default)
        checksum_validation=True,           # MD5 integrity check (default)
    ),
    num_nodes=2,                   # Number of training nodes
    resources_per_node={"gpu": 1}, # Resources per node
    image="pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime",  # Optional image
    packages_to_install=["pyyaml"],  # Extra pip packages
    func_args={"epochs": 100},       # kwargs passed to func
    env={"DEBUG": "1"},              # Environment variables
)
```

## Shared Volume Setup

The LiveTrainer is storage-agnostic. The user or platform must provision and mount the shared volume. Common approaches:

**PodTemplateOverrides (SDK-level):**

```python
from kubeflow.trainer.options.kubernetes import PodTemplateOverrides

options = [
    PodTemplateOverrides(
        target_jobs=["node"],
        volumes=[{"name": "shared", "persistentVolumeClaim": {"claimName": "my-rwx-pvc"}}],
        containers=[{
            "name": "node",
            "volumeMounts": [{"name": "shared", "mountPath": "/mnt/shared"}],
        }],
    ),
]

client.train(trainer=trainer, options=options)
```

**Cluster-level:** Configure the `ClusterTrainingRuntime` to include the volume in its pod template.

**Platform-level:** Some Kubernetes platforms (e.g., Kubeflow Notebooks) automatically mount shared storage into all pods in a namespace.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Updates not detected | `sync_interval` too high | Lower `sync_interval` (e.g., 1 for testing) |
| Partial reads / YAML errors | File written while being read | Enable `checksum_validation` (default) or `use_lock_file` |
| Parameter not applied | Key not in optimizer param groups | Check that the YAML key matches an optimizer key (e.g., use `learning_rate`, not `lr`) |
| Parameter filtered out | Not in `hot_reload_params` whitelist | Add the key to `hot_reload_params`, or set it to `None` to allow all |
| `_kubeflow_sync_loop` not defined | Function called outside LiveTrainer context | These globals are injected by the instrumentation wrapper — they only exist inside a LiveTrainer-submitted TrainJob |

## Running the Existing Tests

The LiveTrainer has comprehensive unit tests that can be run locally:

```bash
# All LiveTrainer tests
uv run pytest kubeflow/trainer/livetrainer/ -v

# Individual test files
uv run pytest kubeflow/trainer/livetrainer/types_test.py -v    # Dataclass validation
uv run pytest kubeflow/trainer/livetrainer/utils_test.py -v    # CRD generation
uv run pytest kubeflow/trainer/livetrainer/runtime_test.py -v  # FileWatcher, SyncLoop, wrapper
```

## Adding Future Examples

To add a new example to this directory:

1. **Create a subdirectory** under `examples/` named after the feature (e.g., `examples/my-feature/`).
2. **Add a Jupyter notebook** (`.ipynb`) as the primary entry point. Structure it with numbered markdown sections, inline assertions for validation, and a clear progression from local testing to cluster submission.
3. **Add a `README.md`** in the subdirectory explaining:
   - What the example demonstrates
   - Prerequisites (Python version, cluster requirements, external services)
   - How to run the notebook
   - Key API surfaces and configuration options
   - Troubleshooting tips for common issues
4. **Keep cluster-dependent cells commented out** so the notebook runs end-to-end locally without a live cluster.
5. **Include inline assertions** (`assert`) in notebook cells so that running all cells serves as a validation pass.
6. **Update the top-level `examples/README.md`** to list the new example.
