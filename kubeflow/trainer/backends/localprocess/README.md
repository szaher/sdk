# Local Process Backend

The Local Process Backend is a lightweight execution backend for the Kubeflow Trainer SDK that runs training jobs as local subprocesses on your machine, eliminating the need for a Kubernetes cluster during development and testing.

## Overview

The Local Process Backend provides a local development environment that mimics the behavior of distributed training systems while running entirely on your local machine. It creates isolated Python virtual environments for each training job, manages dependencies, and executes your training code in controlled subprocesses.

## Key Features

- 🔬 **Local Development**: Run training jobs locally without Kubernetes
- 🏗️ **Isolated Environments**: Each job runs in its own Python virtual environment
- 📦 **Dependency Management**: Automatic installation and conflict resolution of Python packages
- 🔄 **Job Lifecycle**: Complete job management (create, monitor, logs, cancel, cleanup)
- 🧵 **Threaded Execution**: Non-blocking job execution with real-time log streaming
- 🧹 **Automatic Cleanup**: Optional cleanup of virtual environments after completion
- ⚡ **Fast Iteration**: Quick setup and teardown for rapid experimentation

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LocalProcessBackend                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌──────────────────┐                     │
│  │ Training Job  │────│ Virtual Env      │                     │
│  │ (LocalJob)    │    │ /tmp/xyz123...   │                     │
│  │               │    │ ├── bin/python   │                     │
│  │ - Threading   │    │ ├── lib/...      │                     │
│  │ - Log Stream  │    │ └── train_xyz.py │                     │
│  │ - Status Mgmt │    └──────────────────┘                     │
│  └───────────────┘                                             │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────────┤
│  │ Package Management (utils.py)                              │
│  │ - Runtime dependencies (torch, etc.)                       │
│  │ - Trainer dependencies (custom packages)                   │
│  │ - Conflict resolution & merging                            │
│  └─────────────────────────────────────────────────────────────┤
└─────────────────────────────────────────────────────────────────┘
```

## Components

### LocalProcessBackend
The main backend class that implements the `ExecutionBackend` interface:
- **Job Management**: Create, list, get, delete training jobs
- **Runtime Management**: List available runtimes and their packages
- **Log Streaming**: Real-time access to job logs with follow support
- **Status Monitoring**: Wait for job completion with configurable timeouts

### LocalJob
A threaded subprocess wrapper that provides:
- **Non-blocking Execution**: Jobs run in background threads
- **Real-time Logging**: Stream stdout/stderr as jobs execute
- **Process Control**: Start, stop, and monitor subprocess lifecycle
- **Dependency Support**: Wait for dependent jobs to complete

### Virtual Environment Management
Each training job gets its own isolated environment:
- **Temporary Directory**: Unique `/tmp/` directory per job
- **Python Virtual Environment**: Isolated package installation
- **Dependency Resolution**: Smart merging of runtime and trainer packages
- **Automatic Cleanup**: Optional removal of environments after completion

## Supported Runtimes

Currently supported runtimes:

| Runtime | Framework | Description | Packages |
|---------|-----------|-------------|----------|
| `torch-distributed` | PyTorch | Distributed PyTorch training | `torch` |

## Usage

### Basic Setup

```python
from kubeflow.trainer.backends.localprocess.backend import LocalProcessBackend
from kubeflow.trainer.backends.localprocess.types import LocalProcessBackendConfig
from kubeflow.trainer.types import types

# Create backend with configuration
config = LocalProcessBackendConfig(cleanup_venv=True)
backend = LocalProcessBackend(cfg=config)

# List available runtimes
runtimes = backend.list_runtimes()
runtime = backend.get_runtime("torch-distributed")
```

### Training Job Example

```python
# Define your training function
def train_model():
    import torch
    print("Starting PyTorch training...")
    
    # Your training logic here
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    for epoch in range(5):
        # Training step
        loss = torch.nn.functional.mse_loss(
            model(torch.randn(32, 10)), 
            torch.randn(32, 1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Create trainer configuration
trainer = types.CustomTrainer(
    func=train_model,
    func_args=None,  # No arguments needed
    packages_to_install=["torch", "numpy"],
    pip_index_urls=["https://pypi.org/simple"],
    env={"CUDA_VISIBLE_DEVICES": "0"},
)

# Start training job
job_name = backend.train(runtime=runtime, trainer=trainer)
print(f"Started training job: {job_name}")
```

### Job Monitoring

```python
# Get job details
job = backend.get_job(job_name)
print(f"Job Status: {job.status}")
print(f"Steps: {[step.name for step in job.steps]}")

# Stream logs in real-time
print("Training logs:")
for log_line in backend.get_job_logs(job_name, follow=True):
    print(log_line, end='')

# Wait for completion
completed_job = backend.wait_for_job_status(
    job_name, 
    status={constants.TRAINJOB_COMPLETE},
    timeout=600
)
print(f"Training completed with status: {completed_job.status}")
```

### Job Management

```python
# List all jobs
jobs = backend.list_jobs()
for job in jobs:
    print(f"Job: {job.name}, Status: {job.status}")

# List jobs for specific runtime
torch_jobs = backend.list_jobs(runtime=runtime)

# Delete a job (cancels if running)
backend.delete_job(job_name)
```

## Configuration Options

### LocalProcessBackendConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `cleanup_venv` | `bool` | `True` | Automatically delete virtual environments after job completion |

### CustomTrainer Options

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `func` | `Callable` | ✅ | Training function to execute |
| `func_args` | `dict` | ❌ | Arguments to pass to training function |
| `packages_to_install` | `list[str]` | ❌ | Additional Python packages to install |
| `pip_index_urls` | `list[str]` | ❌ | PyPI URLs for package installation |
| `env` | `dict[str, str]` | ❌ | Environment variables for the training process |

## How It Works

### 1. Job Creation
```bash
# Generated unique job name: a1b2c3d4e5f
# Create temporary directory: /tmp/a1b2c3d4e5f_xyz/
# Set up virtual environment with isolation
```

### 2. Environment Setup
```bash
python -m venv --without-pip /tmp/a1b2c3d4e5f_xyz/
source /tmp/a1b2c3d4e5f_xyz/bin/activate
python -m ensurepip --upgrade --default-pip
```

### 3. Dependency Installation
```bash
# Smart package merging - trainer packages override runtime packages
pip install --index-url https://pypi.org/simple torch numpy
```

### 4. Training Code Preparation
```python
# Extract training function source code
# Write to: /tmp/a1b2c3d4e5f_xyz/train_a1b2c3d4e5f.py

def train_model():
    import torch
    print("Starting PyTorch training...")
    # ... your code ...

train_model()  # Auto-generated function call
```

### 5. Execution
```bash
# For PyTorch framework:
/tmp/a1b2c3d4e5f_xyz/bin/torchrun train_a1b2c3d4e5f.py

# For other frameworks:
/tmp/a1b2c3d4e5f_xyz/bin/python train_a1b2c3d4e5f.py
```

### 6. Cleanup (Optional)
```bash
rm -rf /tmp/a1b2c3d4e5f_xyz/
```

## Package Dependency Resolution

The backend implements intelligent package dependency management:

### Rules
1. **Trainer Override**: Trainer packages take precedence over runtime packages
2. **Case-Insensitive Matching**: Package names are normalized (PEP 503)
3. **Duplicate Detection**: Prevents duplicate packages in trainer dependencies
4. **Order Preservation**: Maintains installation order for reproducibility

### Example
```python
# Runtime packages: ["torch==1.9.0", "numpy"]  
# Trainer packages: ["torch==2.0.0", "scipy"]
# Result: ["numpy", "torch==2.0.0", "scipy"]
```

## Error Handling

### Common Issues

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: CustomTrainer must be set` | Using BuiltinTrainer | Use CustomTrainer instead |
| `ValueError: Runtime 'name' not found` | Invalid runtime name | Use `list_runtimes()` to see available options |
| `ValueError: No python executable found` | Missing Python | Install Python or ensure it's in PATH |
| `No TrainJob with name 'name'` | Job doesn't exist | Check job name spelling |

### Job Status Flow

```
Created → Running → Complete
    ↓       ↓
  Failed ← Failed
```

## Development and Testing

### Running Tests
```bash
# Run all tests
pytest kubeflow/trainer/backends/localprocess/backend_test.py -v

# Run specific test
pytest kubeflow/trainer/backends/localprocess/backend_test.py::test_train -v
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable debug logging for detailed execution info
backend = LocalProcessBackend(cfg=config)
```

## Limitations

1. **Single Machine**: Only runs on local machine, no distributed training
2. **CustomTrainer Only**: Does not support BuiltinTrainer configurations  
3. **No GPU Scheduling**: Cannot manage GPU allocation across multiple jobs
4. **Process Isolation**: Jobs are isolated by virtual environment, not containers
5. **Limited Scaling**: Not suitable for large-scale production training

## When to Use

### ✅ Perfect For
- **Local Development**: Developing and testing training code
- **Experimentation**: Quick iteration on training algorithms
- **CI/CD Pipelines**: Automated testing of training workflows
- **Educational Use**: Learning distributed training concepts
- **Prototyping**: Validating ideas before cluster deployment

### ❌ Not Suitable For
- **Production Training**: Large-scale distributed training workloads
- **Multi-Node Training**: Training across multiple machines
- **Resource Management**: Fine-grained GPU/memory allocation
- **Long-Running Jobs**: Training jobs that run for days/weeks
- **High Availability**: Mission-critical training pipelines

## Migration Guide

### From Kubernetes Backend
```python
# Before (Kubernetes)
from kubeflow.trainer.backends.kubernetes.backend import KubernetesBackend
backend = KubernetesBackend(config)

# After (Local Process)
from kubeflow.trainer.backends.localprocess.backend import LocalProcessBackend
from kubeflow.trainer.backends.localprocess.types import LocalProcessBackendConfig
backend = LocalProcessBackend(cfg=LocalProcessBackendConfig())
```

### To Kubernetes Backend
```python
# Ensure your CustomTrainer is compatible
# Test locally first, then deploy to cluster
# Same training code works in both backends!
```

## Contributing

### File Structure
```
kubeflow/trainer/backends/localprocess/
├── backend.py          # Main LocalProcessBackend class
├── job.py             # LocalJob subprocess management  
├── utils.py           # Virtual environment & dependency utilities
├── types.py           # Configuration and data types
├── constants.py       # Templates and runtime definitions
├── backend_test.py    # Comprehensive test suite
└── README.md          # This documentation
```

### Adding New Runtimes
1. Add runtime definition to `constants.py`:
```python
local_runtimes.append(
    base_types.Runtime(
        name="my-framework",
        trainer=types.LocalRuntimeTrainer(
            trainer_type=base_types.TrainerType.CUSTOM_TRAINER,
            framework="my-framework",
            packages=["my-framework-package"],
        ),
    )
)
```

2. Update `utils.py` to handle framework-specific commands
3. Add tests to `backend_test.py`
4. Update this README.md

## Support

For questions, issues, or contributions:
- **Issues**: Create GitHub issues for bugs or feature requests
- **Documentation**: Check the main Kubeflow Trainer documentation
- **Community**: Join Kubeflow community discussions
