# Specialized Trainer Abstractions and RuntimeConfig for the Kubeflow SDK

<!--
This proposal targets the kubeflow/sdk repository.
Directory: docs/proposals/285-specialized-trainers/README.md
-->

|                |                                                              |
| -------------- | ------------------------------------------------------------ |
| **Authors**    | @szaher                                                      |
| **Status**     | Draft                                                        |
| **Created**    | 2026-02-11                                                   |
| **Reviewers**  |                                                              |
| **Supersedes** | N/A                                                          |
| **Relevant Issues** | https://github.com/kubeflow/sdk/issues/285              |

## Table of Contents

<!-- toc -->
- [Specialized Trainer Abstractions and RuntimeConfig for the Kubeflow SDK](#specialized-trainer-abstractions-and-runtimeconfig-for-the-kubeflow-sdk)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Motivation](#motivation)
    - [User Value](#user-value)
    - [Personas](#personas)
    - [Why Specialized Trainers?](#why-specialized-trainers)
    - [Goals](#goals)
    - [Non-Goals](#non-goals)
  - [Current State Analysis](#current-state-analysis)
    - [CustomTrainer](#customtrainer)
    - [BuiltinTrainer](#builtintrainer)
    - [TrainerClient.train()](#trainerclienttrain)
    - [Identified Limitations](#identified-limitations)
  - [Proposal](#proposal)
    - [A. BaseTrainer Abstract Interface](#a-basetrainer-abstract-interface)
    - [B. FuncTrainer — Function-Driven Base](#b-functrainer--function-driven-base)
      - [TorchTrainer](#torchtrainer)
      - [DeepSpeedTrainer](#deepspeedtrainer)
      - [JAXTrainer](#jaxtrainer)
      - [XGBoostTrainer](#xgboosttrainer)
    - [C. ConfigTrainer — Config-Driven Base](#c-configtrainer--config-driven-base)
      - [BuiltinTrainer Migration Path](#builtintrainer-migration-path)
    - [D. RuntimeConfig](#d-runtimeconfig)
    - [E. TrainerClient Changes](#e-trainerclient-changes)
    - [F. Config-Driven LLM Trainers](#f-config-driven-llm-trainers)
      - [Current Coupling to TorchTune](#current-coupling-to-torchtune)
      - [Placement Under BaseTrainer](#placement-under-basetrainer)
      - [Config-Driven Runtime Resolution](#config-driven-runtime-resolution)
      - [Trainer Registry and the Out-of-Tree Extension Path](#trainer-registry-and-the-out-of-tree-extension-path)
      - [Field Ordering Under Inheritance](#field-ordering-under-inheritance)
      - [TorchTuneTrainer](#torchtunetrainer)
      - [TRLTrainer](#trltrainer)
      - [Why TRL as the First In-Tree Config-Driven Framework](#why-trl-as-the-first-in-tree-config-driven-framework)
      - [BuiltinTrainer After This Proposal](#builtintrainer-after-this-proposal)
  - [Design Details](#design-details)
    - [Runtime Auto-Discovery](#runtime-auto-discovery)
    - [Runtime Validation](#runtime-validation)
    - [Trainer Responsibility Boundary](#trainer-responsibility-boundary)
    - [Framework Argument Separation](#framework-argument-separation)
    - [Backend Integration](#backend-integration)
    - [Type Hierarchy Diagram](#type-hierarchy-diagram)
  - [User-Facing API Examples](#user-facing-api-examples)
    - [Before (Current)](#before-current)
    - [After (Proposed)](#after-proposed)
  - [Migration and Backward Compatibility](#migration-and-backward-compatibility)
  - [Test Plan](#test-plan)
    - [Unit Tests](#unit-tests)
    - [Integration Tests](#integration-tests)
    - [Backward Compatibility Tests](#backward-compatibility-tests)
  - [Implementation Plan](#implementation-plan)
  - [Graduation Criteria](#graduation-criteria)
    - [Alpha (target: current cycle)](#alpha-target-current-cycle)
    - [Beta](#beta)
    - [GA](#ga)
  - [Open Questions](#open-questions)
  - [Alternatives Considered](#alternatives-considered)
    - [1. Extend CustomTrainer with a `framework` field instead of new classes](#1-extend-customtrainer-with-a-framework-field-instead-of-new-classes)
    - [2. Use Pydantic `BaseModel` instead of `@dataclass`](#2-use-pydantic-basemodel-instead-of-dataclass)
    - [3. Put RuntimeConfig inside BaseTrainer instead of as a separate parameter](#3-put-runtimeconfig-inside-basetrainer-instead-of-as-a-separate-parameter)
    - [4. Automatic runtime selection with scoring/ranking instead of strict single-match](#4-automatic-runtime-selection-with-scoringranking-instead-of-strict-single-match)
    - [5. Flat hierarchy: all trainers inherit directly from BaseTrainer](#5-flat-hierarchy-all-trainers-inherit-directly-from-basetrainer)
    - [6. Have specialized trainers inherit from CustomTrainer](#6-have-specialized-trainers-inherit-from-customtrainer)
  - [References](#references)
<!-- /toc -->

---

## Overview

This proposal introduces two backward-compatible enhancements to the Kubeflow SDK
(`kubeflow/sdk`) trainer subsystem:

1. **Specialized, framework-aware trainer abstractions** — A three-level type hierarchy:
   `BaseTrainer` (common interface) → `FuncTrainer` (function-driven) and `ConfigTrainer`
   (config-driven), with framework-specific implementations (`TorchTrainer`,
   `DeepSpeedTrainer`, `JAXTrainer`, etc.) that automatically discover and validate the
   correct `ClusterTrainingRuntime` using the `trainer.kubeflow.org/framework` label.
   This fills the "missing middle" between the overly generic `CustomTrainer` and the
   overly narrow `BuiltinTrainer`.

2. **`RuntimeConfig` dataclass** — A dedicated configuration object that cleanly separates
   per-job runtime environment settings (packages, pip config, environment variables) from
   training logic and scaling parameters. This replaces the current pattern where
   `CustomTrainer` conflates runtime concerns with trainer concerns.

Both changes are purely additive. Existing code using `CustomTrainer`, `BuiltinTrainer`, and
`TrainerClient.train()` remains fully functional without modification.

---

## Motivation

### User Value

The Kubeflow Trainer v2 architecture (KEP-2170) introduced a powerful separation between
the *what* (`TrainJob`) and the *how* (`TrainingRuntime` / `ClusterTrainingRuntime`). The
SDK exposes this through `TrainerClient.train()`, which accepts a trainer and an optional
runtime reference. However, the current SDK abstractions create a usability gap:

- **`CustomTrainer`** requires the user to know the runtime name, manually look it up
  via `get_runtime()`, and pass both training arguments and runtime-environment settings
  (packages, pip URLs, env vars) into a single flat dataclass. It provides no
  framework-specific validation or argument handling.

- **`BuiltinTrainer`** is restricted to a single use case (`TorchTuneConfig`) and does
  not accept user-defined training functions.

For the majority of distributed training workloads — "run this PyTorch DDP function on
N nodes" or "run this DeepSpeed training script across a cluster" — neither abstraction
fits well.
Users must either use the low-level `CustomTrainer` with manual runtime wiring, or
fall back to raw YAML.

### Personas

This proposal benefits all three personas defined in KEP-2170:

| Persona | Current Pain | Proposed Improvement |
|---|---|---|
| **Data Scientist / ML Engineer** | Must understand runtime names and Kubernetes concepts to use `CustomTrainer` | Uses `TorchTrainer(func=my_fn)` — runtime is auto-discovered |
| **MLOps Engineer** | Must help data scientists find the correct runtime name for their framework | Framework validation catches mismatches at submission time |
| **Platform Admin / DevOps** | Cannot enforce that users pick the correct runtime for their framework | Trainers validate `trainer.kubeflow.org/framework` labels on runtimes |

### Why Specialized Trainers?

Beyond runtime auto-discovery and framework validation, specialized trainers provide
a set of capabilities that `CustomTrainer` cannot offer:

| Capability | `CustomTrainer` | Specialized Trainer (e.g., `TorchTrainer`) |
|---|---|---|
| **Runtime selection** | User must know and pass the runtime name | Auto-discovered from `trainer.kubeflow.org/framework` label |
| **Framework validation** | None — mismatches fail at execution time | Validated at submission time, before `TrainJob` is created |
| **Typed framework arguments** | Untyped `func_args` dict mixes hyperparams with framework args (`max_restarts`, `deepspeed_config`) | Dedicated typed fields with IDE autocomplete and documentation |
| **Separation of concerns** | Runtime env (`packages_to_install`, `env`), scaling (`num_nodes`), training logic (`func`), and framework args all in one flat dataclass | Training logic in `FuncTrainer`, config in `ConfigTrainer`, runtime env in `RuntimeConfig`, scaling on `BaseTrainer` |
| **IDE/type-checker support** | `func_args: dict` — no autocomplete or type checking | Typed fields — autocomplete, mypy, and docstrings per framework |
| **Extensibility** | Adding framework support requires modifying `CustomTrainer` or creating ad-hoc wrappers | New framework = new subclass of `FuncTrainer` or `ConfigTrainer` |
| **Self-documenting API** | `CustomTrainer(func=..., func_args={"max_restarts": 3})` — unclear what's a hyperparam vs. framework arg | `TorchTrainer(func=..., max_restarts=3)` — intent is clear from the type |

**In summary:** Specialized trainers encode framework knowledge into the type system.
The trainer class itself tells you what framework it targets, what arguments it accepts,
and what runtimes it is compatible with. This shifts errors from runtime to definition
time and makes the SDK self-documenting.

### Goals

1. Define a `BaseTrainer` abstract interface that all trainer implementations satisfy,
   enabling the SDK and backends to handle any trainer polymorphically.
2. Define two intermediate abstract classes — `FuncTrainer` for function-driven
   trainers and `ConfigTrainer` for config-driven trainers — that provide shared
   fields and default implementations for their respective categories.
3. Implement framework-specific function-driven trainers (`TorchTrainer`,
   `DeepSpeedTrainer`, `JAXTrainer`, `XGBoostTrainer`) that auto-discover runtimes
   by the `trainer.kubeflow.org/framework` label and validate runtime compatibility.
4. Provide a clear extension point for community-contributed config-driven trainers
   (e.g., `TorchTuneTrainer`, `UnslothTrainer`, `VeRLTrainer`).
5. Introduce a `RuntimeConfig` dataclass to cleanly separate per-job runtime environment
   settings from training-loop and scaling configuration.
6. Maintain 100% backward compatibility with the existing `CustomTrainer`,
   `CustomTrainerContainer`, `BuiltinTrainer`, and `TrainerClient.train()` APIs.

### Non-Goals

1. **Controller/CRD changes.** This proposal is SDK-only. No changes to the Kubeflow
   Trainer controller, `TrainJob` CRD, or `ClusterTrainingRuntime` CRD are required.
2. **New runtime labels or conventions.** We rely on the existing
   `trainer.kubeflow.org/framework` label already required on all runtimes.
3. **Deprecating `CustomTrainer` or `BuiltinTrainer`.** Both remain supported.
   Specialized trainers are an additional option, not a replacement. `ConfigTrainer`
   is designed as the successor to `BuiltinTrainer` for config-driven trainers,
   but the migration is deferred to a follow-up proposal.
4. **Tier 2 trainer implementations.** This proposal defines the extension mechanism
   and interface. Concrete Tier 2 implementations (TorchTune, Transformers, Unsloth,
   Axolotl) will be proposed in follow-up KEPs.
5. **Changes to the `TrainJobTemplate` dataclass.** Template support for specialized
   trainers can be added incrementally.

---

## Current State Analysis

The following is the current SDK API surface as of `kubeflow-sdk v0.1` (source:
[`kubeflow/trainer/types/types.py`](https://github.com/kubeflow/sdk/blob/main/kubeflow/trainer/types/types.py)).

### CustomTrainer

```python
@dataclass
class CustomTrainer:
    func: Callable
    func_args: Optional[dict] = None
    image: Optional[str] = None
    packages_to_install: Optional[list[str]] = None          # Runtime concern
    pip_index_urls: list[str] = field(                       # Runtime concern
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    num_nodes: Optional[int] = None                          # Scaling concern
    resources_per_node: Optional[dict] = None                # Scaling concern
    env: Optional[dict[str, str]] = None                     # Runtime concern
```

**Issues:**

- Mixes runtime-environment settings (`packages_to_install`, `pip_index_urls`, `env`)
  with scaling/resource settings (`num_nodes`, `resources_per_node`) and training logic
  (`func`, `func_args`).
- No framework awareness. A user can pass a PyTorch training function with a
  DeepSpeed runtime and the SDK will not catch the mismatch until the controller
  rejects the job or, worse, it fails at execution time.
- `func_args` is an untyped `dict` that conflates user hyperparameters with framework
  arguments (e.g., `rdzv_endpoint`, `nnodes`) that the Trainer controller already
  injects via environment variables.

### BuiltinTrainer

```python
@dataclass
class BuiltinTrainer:
    config: TorchTuneConfig
```

- Hardcoded to `TorchTuneConfig`. Cannot be extended to other config-driven frameworks
  without modifying the class itself.

### TrainerClient.train()

```python
def train(
    self,
    runtime: Optional[Union[str, types.Runtime]] = None,
    initializer: Optional[types.Initializer] = None,
    trainer: Optional[
        Union[types.CustomTrainer, types.CustomTrainerContainer, types.BuiltinTrainer]
    ] = None,
    options: Optional[list] = None,
) -> str:
```

- The `trainer` parameter type union must be extended for each new trainer type.
- No concept of runtime auto-discovery: if `runtime` is `None`, it defaults to
  `torch-distributed` regardless of the trainer type.

### Identified Limitations

| # | Limitation | Impact |
|---|---|---|
| 1 | **Missing middle abstraction** | 90% of workloads fall between BuiltinTrainer (too specific) and CustomTrainer (too generic) |
| 2 | **Mixed concerns in CustomTrainer** | Runtime config, scaling config, and training logic are tangled in one dataclass |
| 3 | **No framework validation** | Mismatched trainer/runtime combinations fail late — at execution, not submission |
| 4 | **No framework-specific arguments** | torch-specific args (e.g., `max-restarts`, `monitor-interval`) have no typed home |
| 5 | **BuiltinTrainer is not extensible** | Adding a new config-driven framework requires changing the BuiltinTrainer class |
| 6 | **Flat `func_args` dict** | User hyperparameters mix with framework arguments the controller injects |

---

## Proposal

### A. BaseTrainer Abstract Interface

Introduce an abstract base class that defines the contract all trainers must satisfy.
This enables the SDK, backends, and `TrainerClient` to work with any trainer
polymorphically through a single, stable interface.

`BaseTrainer` holds common fields and methods shared by both function-driven and
config-driven trainers. The training-mode-specific concerns (`func`/`func_args` vs.
`config`) are pushed down to the two intermediate classes described in sections B and C.

```python
# kubeflow/trainer/types/types.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional, ClassVar

@dataclass
class BaseTrainer(ABC):
    """Abstract base class for all specialized trainer implementations.

    Provides common fields for scaling and resource configuration, runtime
    auto-discovery via the `supported_frameworks` class variable, and
    framework validation.

    Subclasses should not inherit from this directly — use `FuncTrainer`
    for function-driven trainers or `ConfigTrainer` for config-driven trainers.

    Class Attributes:
        supported_frameworks: Framework identifiers this trainer supports.
            Must match values of the `trainer.kubeflow.org/framework` label
            on ClusterTrainingRuntime resources. Declared as a tuple (immutable)
            and ordered by preference — the first entry is the preferred framework
            for auto-discovery.

    Args:
        num_nodes: Number of nodes for distributed training.
        resources_per_node: Resource requirements per node (cpu, memory, gpu).
        image: Optional custom container image.
    """

    supported_frameworks: ClassVar[tuple[str, ...]]

    num_nodes: Optional[int] = None
    resources_per_node: Optional[dict] = None
    image: Optional[str] = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "__abstractmethods__", None):
            if not hasattr(cls, "supported_frameworks") or not cls.supported_frameworks:
                raise TypeError(
                    f"{cls.__name__} must define a non-empty "
                    f"'supported_frameworks' class variable"
                )

    @abstractmethod
    def get_framework_args(self) -> dict:
        """Return framework-specific CLI/env arguments that do not overlap
        with arguments injected by the Kubeflow Trainer controller
        (e.g., rdzv_endpoint, nnodes are excluded)."""
        ...

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate that the given runtime is compatible with this trainer.

        The default implementation checks the runtime's framework label against
        `supported_frameworks`. Subclasses may add additional validation.

        Raises:
            ValueError: If the runtime's framework is not in supported_frameworks.
        """
        if runtime.trainer.framework not in self.supported_frameworks:
            raise ValueError(
                f"{type(self).__name__} supports frameworks "
                f"{self.supported_frameworks}, but runtime '{runtime.name}' "
                f"has framework '{runtime.trainer.framework}'"
            )
```

**Design decisions:**

- `supported_frameworks` is a `ClassVar[tuple[str, ...]]` — immutable and class-level.
  It is a property of the trainer *class*, not of individual instances. The tuple is
  ordered by preference: the first entry is the preferred framework for auto-discovery.
  `__init_subclass__` enforces that every concrete subclass defines a non-empty
  `supported_frameworks`, catching missing declarations at class definition time
  rather than at runtime.
- Common fields (`num_nodes`, `resources_per_node`, `image`) live on `BaseTrainer` so
  every trainer inherits them without repetition.
- `get_framework_args()` is the only abstract method on `BaseTrainer`. Training-mode
  concerns (`get_train_func()`, `get_config()`) are defined on the intermediate classes.
- `validate_runtime()` has a default implementation so subclasses get validation for
  free but can extend it.
- Methods use `get_*` naming to clearly indicate they are accessors, not setters.

### B. FuncTrainer — Function-Driven Base

`FuncTrainer` is the base class for trainers where the user provides a Python training
function. It owns the `func` and `func_args` fields and implements the corresponding
accessors, so concrete framework trainers only need to add framework-specific fields.

```python
@dataclass
class FuncTrainer(BaseTrainer):
    """Base class for function-driven trainers.

    The user provides a training function that is serialized and executed
    within the distributed environment configured by the runtime.

    Args:
        func: The training function. Each node executes this function.
        func_args: Arguments passed to the training function. Should contain
            only user hyperparameters — framework arguments like rdzv_endpoint
            and nnodes are injected by the Kubeflow Trainer controller.
    """

    func: Callable
    func_args: Optional[dict] = None

    def get_train_func(self) -> Callable:
        """Return the user-provided training function."""
        return self.func

    def get_train_func_args(self) -> Optional[dict]:
        """Return the arguments to pass to the training function."""
        return self.func_args

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate framework and trainer_type compatibility.

        FuncTrainer requires runtimes with trainer_type == CUSTOM_TRAINER.
        """
        super().validate_runtime(runtime)
        if runtime.trainer.trainer_type != TrainerType.CUSTOM_TRAINER:
            raise ValueError(
                f"{type(self).__name__} requires a runtime with "
                f"trainer_type={TrainerType.CUSTOM_TRAINER.value}, but "
                f"runtime '{runtime.name}' has "
                f"trainer_type={runtime.trainer.trainer_type.value}"
            )
```

Concrete function-driven trainers extend `FuncTrainer` and only need to define
`supported_frameworks`, framework-specific fields, and `get_framework_args()`:

#### TorchTrainer

```python
@dataclass
class TorchTrainer(FuncTrainer):
    """Trainer for PyTorch distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: torch`.

    Args:
        max_restarts: Maximum number of worker group restarts before failing.
            Maps to torchrun --max-restarts.
        monitor_interval: Interval in seconds for the elastic agent to monitor
            workers. Maps to torchrun --monitor-interval.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torch",)

    # Torch-specific arguments (non-overlapping with controller-injected args)
    max_restarts: Optional[int] = None
    monitor_interval: Optional[float] = None

    def get_framework_args(self) -> dict:
        args = {}
        if self.max_restarts is not None:
            args["max-restarts"] = str(self.max_restarts)
        if self.monitor_interval is not None:
            args["monitor-interval"] = str(self.monitor_interval)
        return args
```

#### DeepSpeedTrainer

```python
@dataclass
class DeepSpeedTrainer(FuncTrainer):
    """Trainer for DeepSpeed distributed training workloads.

    DeepSpeed can be bootstrapped via either `torchrun` or `mpirun`, so this
    trainer supports both torch-based and MPI-based runtimes. The SDK
    auto-discovers a compatible runtime by matching the
    `trainer.kubeflow.org/framework` label against the supported frameworks.
    When both runtime types are available, the user must specify the runtime
    explicitly.

    Args:
        deepspeed_config: Path or dict for the DeepSpeed JSON configuration.
            When provided, the config is passed to the DeepSpeed launcher
            via the --deepspeed_config flag.
        num_proc_per_node: Number of processes per node. Maps to
            --num_gpus (DeepSpeed launcher) or --nproc_per_node (torchrun).
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("deepspeed", "torch")

    # DeepSpeed-specific arguments
    deepspeed_config: Optional[Union[str, dict]] = None
    num_proc_per_node: Optional[int] = None

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate framework compatibility and launcher support.

        In addition to the standard framework label check, DeepSpeedTrainer
        verifies that the runtime's launcher is compatible (torchrun or mpirun).
        """
        super().validate_runtime(runtime)
        # TODO: Check runtime.trainer.command for launcher compatibility
        # once the Runtime type exposes launcher metadata.

    def get_framework_args(self) -> dict:
        args = {}
        if self.deepspeed_config is not None:
            if isinstance(self.deepspeed_config, dict):
                import json
                args["deepspeed_config"] = json.dumps(self.deepspeed_config)
            else:
                args["deepspeed_config"] = self.deepspeed_config
        if self.num_proc_per_node is not None:
            args["num-proc-per-node"] = str(self.num_proc_per_node)
        return args
```

#### JAXTrainer

```python
@dataclass
class JAXTrainer(FuncTrainer):
    """Trainer for JAX distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: jax`.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("jax",)

    def get_framework_args(self) -> dict:
        return {}
```

#### XGBoostTrainer

```python
@dataclass
class XGBoostTrainer(FuncTrainer):
    """Trainer for XGBoost distributed training workloads.

    Supports runtimes labeled with `trainer.kubeflow.org/framework: xgboost`.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("xgboost",)

    def get_framework_args(self) -> dict:
        return {}
```

### C. ConfigTrainer — Config-Driven Base

`ConfigTrainer` is the base class for trainers that are driven by a configuration
object rather than a user-provided function. The runtime's entrypoint (e.g.,
`tune run`, `accelerate launch`) handles execution based on the config.

This replaces the current `BuiltinTrainer` pattern with an extensible, `BaseTrainer`-
compatible design that supports runtime auto-discovery and framework validation.

```python
@dataclass
class ConfigTrainer(BaseTrainer):
    """Base class for config-driven trainers.

    Config-driven trainers do not accept a user training function. Instead,
    they accept a configuration object that fully describes the training job.
    The runtime's entrypoint handles execution based on the config.

    Subclasses must implement `get_config()` to return the configuration
    as a dictionary that can be passed to the runtime entrypoint.
    """

    @abstractmethod
    def get_config(self) -> dict:
        """Return the training configuration as a dictionary.

        The returned dict is passed to the runtime entrypoint as arguments
        or mounted as a config file, depending on the backend.
        """
        ...

    def get_framework_args(self) -> dict:
        """Default implementation: delegates to get_config().

        Subclasses may override to separate framework args from config args.
        """
        return self.get_config()

    def validate_runtime(self, runtime: "Runtime") -> None:
        """Validate framework and trainer_type compatibility.

        ConfigTrainer requires runtimes with trainer_type == BUILTIN_TRAINER.
        """
        super().validate_runtime(runtime)
        if runtime.trainer.trainer_type != TrainerType.BUILTIN_TRAINER:
            raise ValueError(
                f"{type(self).__name__} requires a runtime with "
                f"trainer_type={TrainerType.BUILTIN_TRAINER.value}, but "
                f"runtime '{runtime.name}' has "
                f"trainer_type={runtime.trainer.trainer_type.value}"
            )
```

Concrete config-driven trainers extend `ConfigTrainer`. These are **not part of this
proposal's implementation scope** — they will be proposed in follow-up KEPs. The
examples below illustrate the extension pattern:

```python
# Example: future TorchTuneTrainer

@dataclass
class TorchTuneTrainer(ConfigTrainer):
    """Config-driven trainer for TorchTune recipes.

    Replaces the current BuiltinTrainer(config=TorchTuneConfig(...)) pattern.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torch",)

    config: TorchTuneConfig

    def get_config(self) -> dict:
        return self.config.to_dict()
```

```python
# Example: future UnslothTrainer

@dataclass
class UnslothTrainer(ConfigTrainer):
    """Config-driven trainer for Unsloth fine-tuning."""

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torch",)

    model: str
    dataset: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True

    def get_config(self) -> dict:
        return {
            "model": self.model,
            "dataset": self.dataset,
            "max_seq_length": self.max_seq_length,
            "load_in_4bit": self.load_in_4bit,
        }
```

```python
# Example: future VeRLTrainer

@dataclass
class VeRLTrainer(ConfigTrainer):
    """Config-driven trainer for VeRL RLHF training."""

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torch",)

    model: str
    reward_model: str
    algorithm: str = "ppo"

    def get_config(self) -> dict:
        return {
            "model": self.model,
            "reward_model": self.reward_model,
            "algorithm": self.algorithm,
        }
```

#### BuiltinTrainer Migration Path

The existing `BuiltinTrainer` with `TorchTuneConfig` is conceptually a `ConfigTrainer`.
This proposal does **not** deprecate or modify `BuiltinTrainer`, but `ConfigTrainer`
is designed to be its successor:

| Aspect | `BuiltinTrainer` (current) | `ConfigTrainer` subclass (future) |
|---|---|---|
| Runtime discovery | None (hardcoded) | Auto-discovery via `supported_frameworks` |
| Framework validation | None | `validate_runtime()` |
| RuntimeConfig support | No | Yes |
| Extension model | Modify `BuiltinTrainer` class | Create new `ConfigTrainer` subclass |
| Config type | Hardcoded `TorchTuneConfig` | Any config via `get_config()` |

A follow-up proposal will define the concrete migration from `BuiltinTrainer` to
`ConfigTrainer` subclasses for TorchTune and other config-driven frameworks. Until then,
`BuiltinTrainer` remains fully supported and unchanged.

### D. RuntimeConfig

Extract runtime-environment settings from `CustomTrainer` into a dedicated dataclass.
This provides a clean separation of concerns and allows runtime configuration to be
reused across any trainer type.

```python
@dataclass
class RuntimeConfig:
    """Per-job runtime environment configuration.

    Separates runtime-environment concerns (what packages to install, what
    environment variables to set) from training-loop and scaling concerns.

    This is passed to `TrainerClient.train()` and applies regardless of
    the trainer type used. It is a separate parameter on `train()` — not
    embedded in trainers — because runtime configuration is orthogonal to
    both trainer type and initializer. The same RuntimeConfig applies to
    all training pods, and keeping it at the `train()` call site allows
    users to set custom env for initializers in the future without changing
    the trainer classes.

    Args:
        packages_to_install: Python packages to install before running the
            training function (e.g., ["transformers>=4.40", "datasets"]).
        pip_index_urls: PyPI index URLs. The first URL is the primary index;
            remaining URLs are extra indexes.
        env: Environment variables to set in all training nodes.
    """

    packages_to_install: Optional[list[str]] = None
    pip_index_urls: list[str] = field(
        default_factory=lambda: list(constants.DEFAULT_PIP_INDEX_URLS)
    )
    env: Optional[dict[str, str]] = None
```

**Design decisions:**

- Uses `@dataclass` (not Pydantic `BaseModel`) to be consistent with the rest of the
  SDK codebase.
- Field names (`packages_to_install`, `pip_index_urls`) are consistent with the
  existing `CustomTrainer` fields and with KFP's `PipelinesClient`.
- Pip configuration is flattened into `RuntimeConfig` rather than nested in a separate
  `PipConfig` type, keeping the API surface minimal. Additional pip options (e.g.,
  `--quiet`, `--user`) can be added as fields later if needed.
- `RuntimeConfig` is a separate `train()` parameter — not embedded in trainers —
  because runtime configuration is orthogonal to trainer type. The same `RuntimeConfig`
  should be usable with `CustomTrainer`, `FuncTrainer`, or `ConfigTrainer` subclasses.
  It also allows future extension to apply env vars to initializers.
- `RuntimeConfig` is optional — when not provided, the trainer's own fields
  (`packages_to_install`, `env` on `CustomTrainer`) or the runtime defaults are used.
  This preserves backward compatibility.
- **Merge semantics:** When both `RuntimeConfig` and `CustomTrainer` fields are
  provided, `RuntimeConfig` fields override `CustomTrainer` fields **only when the
  `RuntimeConfig` field is not `None`**. For example, if
  `RuntimeConfig(env={"DEBUG": "1"})` is passed alongside
  `CustomTrainer(packages_to_install=["torch"])`, the trainer's `packages_to_install`
  is preserved because `RuntimeConfig.packages_to_install` is `None`. This is a
  field-level merge, not a wholesale replacement.

### E. TrainerClient Changes

The `TrainerClient.train()` method signature is extended to accept the new types:

```python
class TrainerClient:

    def train(
        self,
        runtime: Optional[Union[str, "Runtime"]] = None,
        initializer: Optional["Initializer"] = None,
        trainer: Optional[
            Union[
                "CustomTrainer",
                "CustomTrainerContainer",
                "BuiltinTrainer",
                "BaseTrainer",        # NEW: accepts any specialized trainer
            ]
        ] = None,
        runtime_config: Optional["RuntimeConfig"] = None,  # NEW
        options: Optional[list] = None,
    ) -> str:
```

When a `BaseTrainer` subclass is passed:

1. If `runtime` is `None`, the SDK calls `list_runtimes()` and filters by the
   `trainer.kubeflow.org/framework` label matching the trainer's
   `supported_frameworks`.
2. If exactly one matching runtime is found, it is used automatically.
3. If multiple matching runtimes are found, a `ValueError` is raised listing the
   available options and instructing the user to specify one explicitly.
4. If `runtime` is provided (as a name or `Runtime` object), the trainer's
   `validate_runtime()` method is called to verify compatibility.
5. The backend dispatches based on the trainer type: `FuncTrainer` subclasses are
   handled via `get_train_func()` / `get_train_func_args()`, while `ConfigTrainer`
   subclasses are handled via `get_config()`. Both share `get_framework_args()`.

When `runtime_config` is provided, its values take precedence over any
runtime-environment fields on `CustomTrainer` (for backward compatibility, those
fields remain on `CustomTrainer` but `RuntimeConfig` is the preferred mechanism).

**`runtime=None` behavior by trainer type:**

| Trainer type | `runtime=None` behavior |
|---|---|
| `CustomTrainer` / `BuiltinTrainer` | Defaults to `constants.DEFAULT_TRAINING_RUNTIME` ("torch-distributed"), preserving existing behavior. |
| `BaseTrainer` subclasses (`TorchTrainer`, etc.) | Auto-discovery via `_resolve_runtime()` — finds runtimes matching `supported_frameworks`. |

This distinction is intentional: existing code must not change behavior, while new
trainer types benefit from auto-discovery. The `train()` docstring documents this
difference explicitly.

### F. Config-Driven LLM Trainers

Section C leaves the concrete `ConfigTrainer` subclasses to follow-up work: "Concrete
config-driven trainers extend `ConfigTrainer`. These are **not part of this proposal's
implementation scope** — they will be proposed in follow-up KEPs." This section populates
that branch. It adds no new abstraction to the `BaseTrainer` / `FuncTrainer` /
`ConfigTrainer` hierarchy. It specifies the two concrete config-driven trainers the LLM
post-training use case requires (`TorchTuneTrainer`, `TRLTrainer`), the two SDK internals
that must change so that a config-driven framework other than TorchTune can be expressed at
all, and the registry through which frameworks outside the SDK tree register themselves.

#### Current Coupling to TorchTune

TorchTune is not merely the default config-driven framework; it is the only one the SDK can
represent, and it is hardcoded at four points:

| # | Coupling | Location |
|---|---|---|
| 1 | `BuiltinTrainer.config` is annotated with the concrete `TorchTuneConfig` type | `kubeflow/trainer/types/types.py:226-236` |
| 2 | The framework identifier is derived by reflecting on that annotation: `BuiltinTrainer.__annotations__["config"].__name__.lower().replace("config", "")`, yielding `"torchtune"`. The code's own comment reads "Change it to list: BUILTIN_CONFIGS, once we support more Builtin Trainer configs." | `types.py:239-240` |
| 3 | `trainer_type` and the container entrypoint are both selected by string-comparing the runtime's framework label against that single derived constant | `kubeflow/trainer/backends/kubernetes/utils.py:114-119`, `:140-148` |
| 4 | Config-to-argument translation is guarded by an `isinstance` check against `TorchTuneConfig` and delegates to a TorchTune-only emitter | `utils.py:451-452`, `:473-527` |

Coupling #3 is the one that matters most, and it is the least visible of the four.
`trainer_type` is **not** a field on the Runtime CR and is not read
from it; it is *computed* in the SDK as `BUILTIN_TRAINER if framework == types.TORCH_TUNE
else CUSTOM_TRAINER`. A runtime labelled `trainer.kubeflow.org/framework: trl` therefore
resolves to `CUSTOM_TRAINER` today, and `ConfigTrainer.validate_runtime()` — which requires
`TrainerType.BUILTIN_TRAINER` — would reject it. The same is true of
`RuntimeTrainer.command`, which is synthesized by the `if framework == types.TORCH_TUNE`
chain at `utils.py:140-148` and falls through to `TORCH_COMMAND` (the `CustomTrainer`
function-exec script) for any other framework. Without a change to `get_runtime_trainer()`,
no config-driven trainer other than TorchTune can run, regardless of how the type hierarchy
is arranged. Specifying that change is the substantive content of this section.

The motivation is not hypothetical. Active development on TorchTune was stopped effective
immediately on 15 July 2025 ([meta-pytorch/torchtune#2883](https://github.com/meta-pytorch/torchtune/issues/2883));
no new features were added, and the announced commitment to critical bug fixes and security
patches ran only through the end of 2025. The Kubeflow integration exposes supervised
fine-tuning alone. Preference optimization and reinforcement-learning post-training are not
reachable through the SDK at all.

#### Placement Under BaseTrainer

The function-driven versus config-driven distinction this proposal now encodes was worked out
in the review of this PR: @tariq-hasan framed the placement question for config-driven
post-training trainers, @szaher proposed the `FuncTrainer` / `ConfigTrainer` split that the
current revision adopts, and @astefanutti established that
`trainer.kubeflow.org/framework` should remain the discovery key while users stay able to
bring their own runtimes, frameworks and trainers.

[KEP-2839](https://github.com/kubeflow/trainer/pull/3263) (@NarayanaSabari) is the direct
groundwork for this section: it was the first to write the config-driven side up as a
concrete design. It enumerated the TorchTune coupling points, specified a `command` ClassVar
and a `to_args()` rendering method for trainers whose entrypoint is a framework CLI, and
showed `TorchTuneConfig` becoming a `TorchTuneTrainer` with a backward-compatible
`BuiltinTrainer` alias. The `command` / `to_args()` shape specified below is taken from it.
Its final revision — a direction reached with @tariq-hasan — placed config-driven trainers
under an `LLMTrainer` ABC *parallel* to `BaseTrainer`, on the grounds that forcing them under
a single ABC creates dead methods (`get_train_func()` returning `None`) and Liskov
Substitution Principle violations.

That objection holds against a `BaseTrainer` that itself declares `get_train_func()`. It does
not apply to the hierarchy as specified here, for a reason worth stating precisely.

The problem it names is real: a subclass that inherits `get_train_func()` only to return
`None` has been handed an operation that is meaningless for it, and clients are pushed into
value tests (`if trainer.get_train_func() is None`) instead of type tests. The remedy is to
segregate the interface, and that is exactly what
[Alternative #5](#5-flat-hierarchy-all-trainers-inherit-directly-from-basetrainer) already
records as this proposal's reason for introducing the intermediate layer. It rejects the flat
hierarchy because "Config-driven trainers would carry `get_train_func()` returning `None` —
semantically incorrect and error-prone", and because "Backend dispatch would rely on runtime
checks (`if trainer.get_train_func() is None`) instead of type checks
(`isinstance(trainer, ConfigTrainer)`)".

Once that split exists, a parallel ABC is a remedy for a problem that no longer has a cause.
`func`, `func_args`, `get_train_func()` and `get_train_func_args()` live on `FuncTrainer`.
`BaseTrainer` carries `supported_frameworks`, `num_nodes`, `resources_per_node`, `image`,
`get_framework_args()` and `validate_runtime()` — every one of which a config-driven trainer
genuinely has. Substitutability holds in the sense that matters: every client of
`BaseTrainer` (`_resolve_runtime()`, `_build_trainer_cr()`) invokes only those members and
narrows by `isinstance` to reach mode-specific behavior, so no `BaseTrainer` client can call
an operation a `ConfigTrainer` cannot honor.

One caveat should be conceded rather than glossed. `ConfigTrainer.validate_runtime()`
*narrows* a precondition: it rejects `CUSTOM_TRAINER` runtimes that
`BaseTrainer.validate_runtime()` would accept. This is defensible because the base contract
is "raise if the runtime is incompatible with this trainer," and a subclass refining what
*incompatible* means is a refinement of the contract, not a breach of it — but it is the one
place in the hierarchy where a subclass is stricter than its base, and `FuncTrainer` does the
same thing symmetrically.

The dividend of one root is concrete. `train(trainer=...)` keeps a single union rather than
gaining a second root type in six signatures across five files (`api/trainer_client.py:110-113`,
`backends/base.py:45-47`, `backends/kubernetes/backend.py:279-280` and `:753-756`,
`backends/localprocess/backend.py:76-77`, `backends/container/backend.py:261-262`). Runtime
auto-discovery, `supported_frameworks` preference ordering, and `validate_runtime()` are
implemented once. And the trainer↔runtime compatibility check that today lives in the backend
as an `isinstance` chain (`backends/kubernetes/backend.py:770-789`) is absorbed by
`validate_runtime()`, where it applies uniformly to every trainer, so that
`runtime.trainer.trainer_type` is validated and not only the framework label.

#### Config-Driven Runtime Resolution

`ConfigTrainer` gains one class attribute and one concrete method. Neither is a new abstract
method; `get_config()` remains the only abstraction a subclass must supply.

```python
# kubeflow/trainer/types/types.py

@dataclass(kw_only=True)
class ConfigTrainer(BaseTrainer):
    """Base class for config-driven trainers. (Additions to section C.)"""

    command: ClassVar[tuple[str, ...]] = ()

    def to_args(self, initializer: Optional["Initializer"] = None) -> list[str]:
        """Render the config as entrypoint arguments.

        The default renders each entry of ``get_config()`` as a flat
        ``key=value`` override. Frameworks whose CLI uses a different convention
        override this method; TorchTune's overrides are nested
        (``model.lora_rank=...``), so `TorchTuneTrainer` overrides it too and
        delegates to the existing emitter.

        Args:
            initializer: The job's initializer, when the framework's arguments
                depend on where the dataset or model was staged.
        """
        return [f"{key}={value}" for key, value in self.get_config().items()]
```

**Design decisions:**

- **`command` is a `ClassVar` on the trainer, not a constant in `constants.py`.** The
  entrypoint of a config-driven job is a property of the framework's CLI (`("tune", "run")`,
  `("trl",)`), and the trainer class is the only place that knows it. This retires the
  `if framework == types.TORCH_TUNE` entrypoint chain at `utils.py:140-148` for the
  config-driven path: `get_runtime_trainer()` looks the framework label up in the registry
  below and calls `set_command(trainer_cls.command)`. `constants.TORCH_TUNE_COMMAND`
  (`constants.py:179`), whose only consumer is that branch (`utils.py:142`), is deleted with
  it. The `FuncTrainer` path
  (`TORCH_COMMAND` / `MPI_COMMAND` / `DEFAULT_COMMAND`, selected from `ml_policy`) is
  untouched.
- **`trainer_type` is derived from the registry, not from a framework constant.**
  `get_runtime_trainer()` assigns `TrainerType.BUILTIN_TRAINER` when the runtime's framework
  label is claimed by a registered `ConfigTrainer` subclass, and `TrainerType.CUSTOM_TRAINER`
  otherwise. This replaces `utils.py:114-119` and deletes the reflection-derived
  `types.TORCH_TUNE` constant (`types.py:239-240`) along with its only consumers. No new
  runtime label, annotation, or CRD field is introduced: `trainer.kubeflow.org/framework`
  remains the sole discovery key, honoring [Non-Goal #2](#non-goals).
- **`to_args()` takes the initializer.** Today's TorchTune emitter derives
  `dataset.data_files=` / `dataset.data_dir=` from the Hugging Face dataset initializer
  (`utils.py:502-517`). A rendering method that could not see the initializer would silently
  lose that behavior.
- **Rendering moves to the trainer, which amends [Backend Integration](#backend-integration).**
  The `ConfigTrainer` branch of `_build_trainer_cr` becomes
  `trainer_cr.command = list(runtime.trainer.command)` and
  `trainer_cr.args = trainer.to_args(initializer)`, replacing the inline
  `[f"{k}={v}" for k, v in trainer.get_config().items()]`. This is a one-time change to the
  backend, and it is what makes the proposal's stated property — that adding a *further*
  trainer requires no backend changes — true for frameworks whose CLI does not use `key=value`
  overrides. `get_framework_args()` retains the meaning section C gives it: framework
  arguments merged into the rendered config; the default `to_args()` renders
  `get_config()`, which for `ConfigTrainer` already defaults to the same dictionary.

#### Trainer Registry and the Out-of-Tree Extension Path

The requirement @astefanutti stated in this PR's review is that the SDK stay extensible —
"users can bring their own runtimes / frameworks / trainers" — with
`trainer.kubeflow.org/framework` kept as the discovery key, and @andreyvelich's follow-up
asked how new LLM fine-tuning framework backends would be registered dynamically. The SDK has
no mechanism for this today: backends are wired by a hardcoded
`isinstance` chain (`api/trainer_client.py:56-63`), `kubeflow/trainer/backends/__init__.py`
is empty, and `pyproject.toml` declares no entry points.

The registry introduced here is deliberately narrow. It maps a **framework label value** to a
**`ConfigTrainer` subclass**, and it exists to serve exactly one lookup: the one performed by
`get_runtime_trainer()` above, which must know whether a given framework label is
config-driven and, if so, what its entrypoint is. It is not an execution backend. Dispatch of
a trainer the user has already constructed remains polymorphic — a third party ships
`class MyTrainer(ConfigTrainer)`, the user imports it and passes it to `train()`, and no
lookup is required.

```python
# kubeflow/trainer/types/registry.py

from importlib.metadata import entry_points
from typing import Optional

_CONFIG_TRAINERS: dict[str, type["ConfigTrainer"]] = {}
_discovered = False


def register_config_trainer(cls: type["ConfigTrainer"]) -> type["ConfigTrainer"]:
    """Claim the framework labels declared in `cls.supported_frameworks`."""
    if not cls.supported_frameworks:
        raise ValueError(f"{cls.__name__} must declare supported_frameworks")
    if not cls.command:
        raise ValueError(f"{cls.__name__} must declare a command")
    for framework in cls.supported_frameworks:
        _CONFIG_TRAINERS[framework] = cls
    return cls


def get_config_trainer(framework: str) -> Optional[type["ConfigTrainer"]]:
    """Return the ConfigTrainer claiming this framework label, or None.

    Third-party trainers are discovered lazily, on the first lookup that misses
    the in-tree registrations.
    """
    # Imported inside the function: `types.py` imports `register_config_trainer`
    # from this module, so a module-scope import here would be a cycle.
    from kubeflow.trainer.types.types import ConfigTrainer

    global _discovered
    if framework not in _CONFIG_TRAINERS and not _discovered:
        _discovered = True
        for entry_point in entry_points(group="kubeflow.trainer.config_trainers"):
            candidate = entry_point.load()
            if not (isinstance(candidate, type) and issubclass(candidate, ConfigTrainer)):
                raise ValueError(
                    f"Entry point '{entry_point.name}' in group "
                    f"'kubeflow.trainer.config_trainers' must resolve to a "
                    f"ConfigTrainer subclass, got {candidate!r}"
                )
            register_config_trainer(candidate)
    return _CONFIG_TRAINERS.get(framework)
```

**Design decisions:**

- **Prior art.** @krishdef7 proposed a registry of this shape in this PR's review
  (`@register_backend(TRLConfig)`), keyed on the trainer's config type. This section keys the
  registry on the framework label instead, because the lookup that needs it —
  `get_runtime_trainer()` — starts from a Runtime CR and has only the label to go on; the
  config type is not in scope at that point.
- **The registry is keyed on the label value, not on the label's storage.** If
  `trainer.kubeflow.org/framework` is later promoted from a label to a Runtime API spec
  field, the change is confined to the single accessor that supplies `framework` to
  `get_runtime_trainer()`.
- **Discovery is lazy and fails loudly.** The `importlib.metadata` scan is paid only on a
  registry miss, and a miss after discovery returns `None`, which `get_runtime_trainer()`
  reads as "this runtime is function-driven" — the existing default. An entry point that
  does not resolve to a `ConfigTrainer` subclass raises rather than being silently ignored.
- **Registration is last-writer-wins on the framework label.** An out-of-tree trainer whose
  `supported_frameworks` claims an in-tree label shadows the in-tree class for that label.
  That is the ambient entry-point convention, and it is deliberate: it lets a community fork
  of a stalled framework replace the in-tree trainer without an upstream commit. It is not a
  security boundary — installing a plugin is a trust decision. Raising on a duplicate claim
  was rejected because the collision surfaces inside the discovery scan, which would turn one
  conflicting plugin into a hard failure of every registry lookup, including lookups for
  unrelated labels.
- **In-tree and out-of-tree trainers use the same mechanism.** `TorchTuneTrainer` and
  `TRLTrainer` are registered with `@register_config_trainer`; a third party declares an
  entry point in the `kubeflow.trainer.config_trainers` group and `pip install`s. This is
  the mechanism behind the Graduation Criteria item "Community has contributed at least one
  Tier 2 `ConfigTrainer` subclass," and it is what makes that criterion reachable without an
  upstream commit. The natural install story reuses the existing extras convention
  (`docker`, `podman`, `spark`, `hub`): `pip install kubeflow[trl]`.
- **In-tree registration runs at import time.** `@register_config_trainer` populates the
  registry as a side effect of importing the module that defines the trainer.
  `TorchTuneTrainer` lives in `types.py`, which `utils.py` already imports, so it registers
  unconditionally. `TRLTrainer` lives in a new module, so `kubeflow/trainer/__init__.py` must
  export it: without that export, a process that never imports it — `list_runtimes()` on a
  fresh client, say — finds no claimant for the `trl` label and classifies the runtime as
  function-driven.
- **LlamaFactory is the reference out-of-tree implementation, not a second in-tree backend.**
  LlamaFactory is built on the Hugging Face `Trainer` and PEFT — the same engine TRL uses — so
  an in-tree integration would add a second surface without adding capability. Implementing it
  as an external plugin is the proof that the extension path works with no upstream
  cooperation.
- **Python floor.** `entry_points(group=...)` selection and `@dataclass(kw_only=True)` both
  require Python 3.10, which the SDK already declares (`requires-python = ">=3.10"`).

#### Field Ordering Under Inheritance

The whole hierarchy — `BaseTrainer`, `FuncTrainer`, `ConfigTrainer` and every concrete
trainer — is declared `@dataclass(kw_only=True)`. This amends the illustrative blocks in
sections A–C, which show a bare `@dataclass`: read every new trainer decorator there as
`@dataclass(kw_only=True)`. `BaseTrainer` declares `num_nodes`, `resources_per_node` and
`image` with defaults, so without `kw_only` no subclass could declare a non-defaulted field —
`FuncTrainer.func` and `TorchTuneTrainer.config` would both raise `TypeError: non-default
argument follows default argument` at class-definition time. The alternative (defaulting every
field to `None` and enforcing required-ness at runtime) would forfeit the static-analysis
benefit that motivates typed trainers. `CustomTrainer`, `BuiltinTrainer` and `TorchTuneConfig`
keep their bare `@dataclass` declarations: their construction signatures are public API and do
not change.

#### TorchTuneTrainer

`TorchTuneTrainer` is not new capability. It is the mechanical replacement for the four
TorchTune coupling points listed at the top of this section, and removing that coupling
requires it. Together with `TRLTrainer` it supersedes the deferral in
[Non-Goal #4](#non-goals) for these two frameworks only; Transformers, Unsloth and Axolotl
remain out of scope and stay reachable through the registry above.

```python
# kubeflow/trainer/types/types.py
# Additional imports: `from dataclasses import asdict`,
# `from kubeflow.trainer.types import torchtune`,
# `from kubeflow.trainer.types.registry import register_config_trainer`

@register_config_trainer
@dataclass(kw_only=True)
class TorchTuneTrainer(ConfigTrainer):
    """Config-driven trainer for TorchTune recipes.

    The preferred form of `BuiltinTrainer(config=TorchTuneConfig(...))`.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("torchtune",)
    command: ClassVar[tuple[str, ...]] = ("tune", "run")

    config: TorchTuneConfig

    def get_config(self) -> dict:
        return asdict(self.config)

    def to_args(self, initializer: Optional["Initializer"] = None) -> list[str]:
        """Preserve the existing TorchTune override rendering exactly."""
        return torchtune.get_args_using_torchtune_config(self.config, initializer)
```

**Design decisions:**

- **`supported_frameworks` is `("torchtune",)`, refining section C's sketch.** Section C
  sketches `("torch",)`, but the runtime label value matched by the SDK today is `torchtune`
  (`types.TORCH_TUNE == "torchtune"`, `types.py:239-240`, compared against the
  `trainer.kubeflow.org/framework` label defined at `constants.py:61-62`). With `("torch",)`,
  auto-discovery would match the `torch-distributed` runtime, which is function-driven, and
  `validate_runtime()` would then reject it.
- **`to_args()` delegates to the existing emitter rather than to the default `key=value`
  renderer.** `get_args_from_peft_config` (`utils.py:530-559`) maps `LoraConfig` fields
  through a `field_map` onto `model.*` TorchTune keys and renders
  `model.lora_attn_modules=[...]`; a flat `key=value` walk over `asdict()` would emit
  `lora_rank=8` instead of `model.lora_rank=8` and produce a job that fails. Behavior is
  preserved byte-for-byte by reusing the emitter. The three emitters
  (`get_args_using_torchtune_config`, `get_args_from_peft_config`,
  `get_args_from_dataset_preprocess_config`, `utils.py:473-601`) are moved verbatim from
  `kubeflow/trainer/backends/kubernetes/utils.py` to `kubeflow/trainer/types/torchtune.py`.
  They contain no Kubernetes-specific logic, and leaving them in the backend would make
  `types.py` import a backend module — a cycle, since `utils.py` already imports `types`.
- **Duplicate scaling fields.** `TorchTuneConfig` carries its own `num_nodes` and
  `resources_per_node` (`types.py:191-222`), which now also exist on `BaseTrainer`. The
  merge happens in `_build_trainer_cr`, which already reads both fields off `BaseTrainer`:
  the fields on the trainer take precedence, and the `TorchTuneConfig` duplicates are promoted
  only when the trainer's are unset. Neither field is ever rendered into `args` — the TorchTune
  emitter does not emit them today, and `TorchTuneTrainer.to_args()` reuses it verbatim. The
  duplicates follow the same schedule as `BuiltinTrainer`: no change in Alpha, `FutureWarning`
  in Beta, formal deprecation at GA.

#### TRLTrainer

```python
# kubeflow/trainer/types/trl.py

from dataclasses import dataclass, fields
from enum import Enum
from typing import ClassVar, Optional

from kubeflow.trainer.types.registry import register_config_trainer
from kubeflow.trainer.types.types import BaseTrainer, ConfigTrainer, Initializer


class TRLMethod(Enum):
    """Post-training method; the value is the TRL CLI subcommand."""

    SFT = "sft"
    DPO = "dpo"
    GRPO = "grpo"


@register_config_trainer
@dataclass(kw_only=True)
class TRLTrainer(ConfigTrainer):
    """Config-driven trainer for Hugging Face TRL post-training.

    The runtime entrypoint is the TRL CLI, which forwards to `accelerate launch`.

    Raises:
        ValueError: If a field is set that does not apply to the selected method.
    """

    supported_frameworks: ClassVar[tuple[str, ...]] = ("trl",)
    command: ClassVar[tuple[str, ...]] = ("trl",)

    method: TRLMethod
    model_name_or_path: str
    dataset_name: str

    learning_rate: Optional[float] = None
    num_train_epochs: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    bf16: Optional[bool] = None

    use_peft: Optional[bool] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    lora_target_modules: Optional[list[str]] = None

    beta: Optional[float] = None                    # DPO, GRPO
    num_generations: Optional[int] = None           # GRPO
    reward_funcs: Optional[list[str]] = None        # GRPO

    # Fields valid only for a subset of methods. Fields absent from this map
    # are valid for every method.
    _METHOD_SCOPED_FIELDS: ClassVar[dict[str, frozenset[TRLMethod]]] = {
        "beta": frozenset({TRLMethod.DPO, TRLMethod.GRPO}),
        "num_generations": frozenset({TRLMethod.GRPO}),
        "reward_funcs": frozenset({TRLMethod.GRPO}),
    }

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Enforce method-scoped field usage.

        Raises:
            ValueError: If a field is set for a method it does not apply to, or
                if a field the selected method requires is unset.
        """
        for name, methods in self._METHOD_SCOPED_FIELDS.items():
            if getattr(self, name) is not None and self.method not in methods:
                allowed = ", ".join(sorted(m.value for m in methods))
                raise ValueError(
                    f"'{name}' applies only to method={allowed}, but "
                    f"method={self.method.value} was selected"
                )
        if self.method is TRLMethod.GRPO and not self.reward_funcs:
            raise ValueError("method=grpo requires at least one entry in 'reward_funcs'")

    def get_config(self) -> dict:
        """Return the TRL CLI options, omitting unset and non-CLI fields."""
        excluded = {"method"} | {f.name for f in fields(BaseTrainer)}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in excluded and getattr(self, f.name) is not None
        }

    def to_args(self, initializer: Optional["Initializer"] = None) -> list[str]:
        """Render as `[<subcommand>, --flag, value, ...]` for the TRL CLI."""
        args: list[str] = [self.method.value]
        for key, value in self.get_config().items():
            if value is True:
                args.append(f"--{key}")           # store_true flag
            elif isinstance(value, list):
                args.append(f"--{key}")
                args.extend(str(item) for item in value)
            else:
                args.append(f"--{key}")
                args.append(str(value))
        return args
```

`TRLTrainer(method=TRLMethod.SFT, model_name_or_path="Qwen/Qwen2.5-0.5B",
dataset_name="stanfordnlp/imdb")` renders to:

```python
trainer_cr.command = ["trl"]                       # from TRLTrainer.command
trainer_cr.args = [
    "sft",
    "--model_name_or_path", "Qwen/Qwen2.5-0.5B",
    "--dataset_name", "stanfordnlp/imdb",
]
```

**Design decisions:**

- **Framework arguments land in `.spec.trainer.args`.** They are produced entirely by
  `ConfigTrainer.to_args()`, against a `.spec.trainer.command` taken from the trainer's
  `command` `ClassVar`. There is no interaction with `MLPolicy`-injected arguments on this
  path, because a config-driven runtime's entrypoint is the framework CLI rather than a
  launcher the control plane parameterizes; the SDK contributes no launch flags of its own.
  Scaling is unaffected: `num_nodes` and `resources_per_node` are written to
  `.spec.trainer.numNodes` / `.spec.trainer.resourcesPerNode` by `_build_trainer_cr` and are
  never rendered into `args`, which is the contract `tune run` already relies on. Translating
  that topology into the environment `accelerate launch` reads is the `trl` runtime image's
  responsibility, not the SDK's.
- **`to_args()` does not consume the initializer.** `model_name_or_path` and `dataset_name`
  are passed through verbatim, so the `trl` CLI resolves them itself. TorchTune's emitter
  rewrites `dataset.data_dir=` / `dataset.data_files=` from a `HuggingFaceDatasetInitializer`
  (`utils.py:502-517`) because the recipe reads a local directory; there is no equivalent
  rewrite for the model on any path today, and `constants.MODEL_PATH` is never emitted into
  trainer arguments. Combining `TRLTrainer` with a Hugging Face initializer therefore stages
  artifacts the `trl` CLI does not read. Wiring `constants.DATASET_PATH` / `constants.MODEL_PATH`
  into TRL's flags depends on the `trl` runtime image's entrypoint contract, which this
  proposal does not specify; it is deferred with that runtime.
- **`LoraConfig` is deliberately not reused.** `LoraConfig` (`types.py:152-187`) is
  TorchTune-shaped: `apply_lora_to_mlp`, `apply_lora_to_output`, `quantize_base`, `use_dora`,
  and module names drawn from TorchTune's model definitions. TRL's PEFT surface is
  `--use_peft`, `--lora_r`, `--lora_alpha`, `--lora_target_modules`; `apply_lora_to_output`
  and `quantize_base` have no TRL analogue. Sharing the type across two frameworks with
  non-overlapping semantics would require a lossy translation layer, so the TRL flags are
  declared directly on `TRLTrainer`.
- **This answers [Open Question #2](#open-questions).** Rather than making `get_config()`
  return an opaque typed object, the trainer itself is the typed object: every TRL flag is a
  statically checked field with IDE completion, semantic constraints the type system cannot
  express are enforced in `validate()` at construction time, and `get_config()` stays the
  plain-`dict` seam the rendering path consumes.
- **Unsloth is a `TRLTrainer` concern, not a sibling class.** Unsloth is an acceleration
  layer over the Hugging Face `Trainer`: its models are passed directly to TRL's `SFTTrainer`
  and `DPOTrainer`. Modelling it as the `ConfigTrainer` sibling section C sketches would
  duplicate the entire TRL field set to express a runtime optimization. It belongs in the
  runtime image: an Unsloth-accelerated image is still labelled
  `trainer.kubeflow.org/framework: trl`, so it is discovered by the same registry entry,
  validated by the unmodified `TRLTrainer`, and selected with `train(runtime=...)`. No SDK
  field is added — `get_config()` renders every non-`None` field as a CLI flag, and
  `--use_unsloth` is not one.

#### Why TRL as the First In-Tree Config-Driven Framework

| Framework | Post-training methods | Maintenance | Entrypoint |
|---|---|---|---|
| TRL | `sft`, `dpo`, `grpo`, `kto`, `reward`, `rloo` on the stable CLI; PPO moved to `trl.experimental` and removed from `trl.trainer` in 0.29 ([huggingface/trl#4466](https://github.com/huggingface/trl/issues/4466)) | Actively maintained by Hugging Face | `trl <method> --flag value`, or `--config <yaml>`; forwards to `accelerate launch` |
| TorchTune | SFT only, in the Kubeflow integration | Active development stopped 15 July 2025; critical fixes committed only through end of 2025 ([#2883](https://github.com/meta-pytorch/torchtune/issues/2883)) | `tune run` |
| LlamaFactory | Broad, but built on the Hugging Face `Trainer` and PEFT | Active | `llamafactory-cli train <config.yaml>` |
| Axolotl | Broad (SFT, DPO, KTO, GRPO), but built on the Hugging Face `Trainer` and PEFT | Active | `axolotl train <config.yaml>` — file-first, so the recipe must be materialized into the container |
| Unsloth | None of its own; an acceleration layer | Active | No independent CLI |

TRL is selected on three grounds. Its stable CLI covers the methods the SDK cannot express at
all today: preference optimization (DPO) and group-relative policy optimization (GRPO). It is
the substrate rather than a wrapper — LlamaFactory and Axolotl are both built on the Hugging
Face `Trainer` and PEFT, so an in-tree TRL trainer adds capability where an in-tree wrapper
would add a second surface over the same engine, which is why LlamaFactory is the reference
out-of-tree plugin above. And its CLI takes flags (`trl <subcommand> --flag value`), so it maps
onto `TrainJob` `command` and `args` with no adapter and no YAML recipe to materialize into the
container; a file-first CLI (`axolotl train <config.yaml>`, `llamafactory-cli train
<config.yaml>`) would need a ConfigMap or a volume. `TRLTrainer(method=TRLMethod.GRPO, ...)`
therefore replaces an untyped YAML recipe with a statically checked dataclass — a benefit
`FuncTrainer` subclasses cannot offer, since there the training code is the user's own and the
trainer's contribution is runtime compatibility.

**Risk:** TRL's surface is not frozen. PPO's relocation to `trl.experimental` shows that
trainers migrate between the stable and experimental namespaces across minor releases, and a
typed SDK dataclass that mirrors TRL flags will drift. Three properties bound the damage.
`TRLTrainer` targets the CLI, not the Python API, and re-implements no TRL semantics — it is
a flag renderer, so a TRL upgrade changes the runtime image rather than the SDK type. It
exposes only `TRLMethod` members that are on TRL's stable CLI, and widening coverage to `kto`,
`reward` or `rloo` is an enum member plus fields. And the registry confines any framework's
churn to that framework's trainer class: TRL's instability cannot reach TorchTune, and neither
can reach `FuncTrainer`.

#### BuiltinTrainer After This Proposal

`BaseTrainer` is the foundation for the builtins. The public `BuiltinTrainer` API is
preserved exactly and requires no user-visible change; the internals it depends on are the
four coupling points enumerated at the top of this section, and all four are replaced.

| Aspect | Before | After |
|---|---|---|
| User entry point | `BuiltinTrainer(config=TorchTuneConfig(...))` | Unchanged and fully supported; `TorchTuneTrainer(config=TorchTuneConfig(...))` is the preferred form |
| Dispatch of `BuiltinTrainer` | `isinstance` chain in the backend (`backend.py:770-789`) | `TrainerClient.train()` converts it to `TorchTuneTrainer(config=trainer.config)` and dispatches through the `ConfigTrainer` path |
| Framework identifier | Reflected from the `config` annotation (`types.py:239-240`) | `supported_frameworks` `ClassVar`; the reflection-derived `types.TORCH_TUNE` constant is deleted, so no second source of truth remains |
| `trainer_type` selection | `framework == types.TORCH_TUNE` (`utils.py:114-119`) | Registry lookup on the framework label |
| Entrypoint selection | `if framework == types.TORCH_TUNE` chain (`utils.py:140-148`) | `ConfigTrainer.command` `ClassVar` |
| Config→args translation | `isinstance(trainer.config, TorchTuneConfig)` guard and a TorchTune-only emitter (`utils.py:451-452`, `:473-527`) | `ConfigTrainer.to_args()`; `TorchTuneTrainer` reuses the same emitter, so output is identical |
| trainer↔runtime validation | Backend (`backend.py:770-789`) | `ConfigTrainer.validate_runtime()` |
| `TorchTuneConfig`, `LoraConfig`, `TorchTuneInstructDataset` | Exported from `kubeflow.trainer` | Unchanged, still exported |
| Adding a framework | Edit `types.py` and `utils.py` | Subclass `ConfigTrainer`, register it, in-tree or via entry point |

The deprecation schedule is the one already stated in
[BuiltinTrainer Migration Path](#builtintrainer-migration-path) and
[Migration and Backward Compatibility](#migration-and-backward-compatibility): no change in
Alpha, a `FutureWarning` pointing at `TorchTuneTrainer` in Beta, formal deprecation at GA.
Every existing TorchTune program continues to run unmodified and produces byte-identical
`TrainJob` arguments.

A runtime labelled `trainer.kubeflow.org/framework: trl` is a prerequisite for `TRLTrainer`
and is owned by the Trainer repository; this proposal does not specify it. The SDK-side
contract is only that the label exists and that the runtime's container entrypoint is the
`trl` CLI.

---

## Design Details

### Runtime Auto-Discovery

The auto-discovery logic lives in the `TrainerClient` (not in the backend), ensuring
consistent behavior across all backends:

```python
def _resolve_runtime(
    self,
    trainer: BaseTrainer,
    runtime: Optional[Union[str, Runtime]],
) -> Runtime:
    """Resolve the runtime for a specialized trainer.

    If runtime is provided, validate it. If not, auto-discover by framework label.
    """
    if runtime is not None:
        # Explicit runtime — validate compatibility
        if isinstance(runtime, str):
            runtime = self.get_runtime(runtime)
        trainer.validate_runtime(runtime)
        return runtime

    # Auto-discover: find runtimes matching the trainer's frameworks.
    # Iterate supported_frameworks in declaration order (most preferred first)
    # to provide deterministic selection when exactly one runtime matches
    # the most-preferred framework.
    all_runtimes = self.list_runtimes()
    matching = []
    for framework in trainer.supported_frameworks:
        matching = [
            r for r in all_runtimes
            if r.trainer.framework == framework
        ]
        if matching:
            break

    if len(matching) == 0:
        raise ValueError(
            f"No runtime found for frameworks {trainer.supported_frameworks}. "
            f"Available runtimes: {[r.name for r in all_runtimes]}"
        )
    if len(matching) > 1:
        raise ValueError(
            f"Multiple runtimes found for framework "
            f"'{matching[0].trainer.framework}': "
            f"{[r.name for r in matching]}. "
            f"Please specify the runtime explicitly."
        )

    return matching[0]
```

**Multi-runtime selection strategy:**

Auto-discovery iterates `supported_frameworks` in declaration order (most preferred
first). For each framework, it collects matching runtimes. If exactly one runtime
matches the most-preferred framework, it is selected automatically. If multiple
runtimes match the same framework, the SDK raises a `ValueError` listing the
available options. If no runtimes match the most-preferred framework, discovery
falls through to the next framework in the tuple.

For example, `DeepSpeedTrainer` declares `supported_frameworks = ("deepspeed", "torch")`.
On a cluster with only a `torch-distributed` runtime, auto-discovery falls through
`"deepspeed"` (no match) and selects the `torch-distributed` runtime. On a cluster
with both a `deepspeed-mpi` and a `torch-distributed` runtime, auto-discovery finds
`deepspeed-mpi` first (matching `"deepspeed"`) and selects it without ambiguity.

When multiple runtimes match the same framework, the user resolves the ambiguity by
passing the `runtime` parameter to `train()`:

```python
# Two torch runtimes exist: "torch-distributed" and "torch-elastic"
# Auto-discovery raises ValueError listing both options.

# User resolves by specifying explicitly:
client.train(
    runtime="torch-elastic",
    trainer=TorchTrainer(func=my_fn, num_nodes=4),
)
```

This is a deliberate design choice. The `runtime` parameter on `train()` is the
single, existing mechanism for runtime selection. Adding a `runtime_name` to
`RuntimeConfig` or to trainer classes would conflate concerns — `RuntimeConfig` is for
packages and environment, trainers are for training logic, and runtime selection belongs
to the `train()` call site. See also
[Alternative #4](#4-automatic-runtime-selection-with-scoringranking-instead-of-strict-single-match)
for why priority-based scoring was rejected.

### Runtime Validation

Validation happens at three levels:

1. **Framework label check** (in `BaseTrainer.validate_runtime()`): Ensures the
   runtime's `trainer.kubeflow.org/framework` label value is in the trainer's
   `supported_frameworks` list.

2. **Trainer type check** (in `FuncTrainer.validate_runtime()` and
   `ConfigTrainer.validate_runtime()`): Ensures the runtime's `trainer_type` matches
   the trainer category. `FuncTrainer` requires `TrainerType.CUSTOM_TRAINER`;
   `ConfigTrainer` requires `TrainerType.BUILTIN_TRAINER`. This catches mismatches
   such as passing a function-driven trainer to a config-only runtime.

3. **Framework-specific checks** (in concrete trainer overrides): For example,
   `DeepSpeedTrainer` could verify that the runtime's launcher configuration
   (torchrun vs. mpirun) is compatible with the selected runtime.

**Validation strategy — SDK vs. control plane:**

SDK validation raises `ValueError` at submission time, *before* the `TrainJob` CR is
created in the cluster. This is intentionally a hard fail, not a warning, because:

1. **Fast feedback.** A `ValueError` with a clear message is immediate. A warning
   that the user ignores leads to a `TrainJob` that fails minutes later in the
   controller or at execution time, wasting cluster resources.
2. **Deterministic checks.** The SDK validates against concrete, known properties
   (framework label, `trainer_type`) — not heuristics. These checks are
   authoritative at the SDK level.
3. **Control plane remains the final arbiter.** The controller's webhook may enforce
   additional constraints (resource quotas, policy, version compatibility) that the
   SDK does not know about. SDK validation is a *subset* of control-plane validation,
   not a replacement.
4. **Overridable.** Subclasses can override `validate_runtime()` to relax or extend
   validation for custom use cases.

In summary: the SDK fails fast on checks it *can* perform (framework, trainer_type),
and defers to the controller for checks it *cannot* perform (quotas, policies).

### Trainer Responsibility Boundary

Trainers are **data objects**, not builders. They expose structured data about the
training job; they do not construct the `TrainJob` CRD, build container entrypoints,
or interact with the Kubernetes API. The responsibility boundary is:

| Concern | Owner | Rationale |
|---|---|---|
| Training function / config | **Trainer** (`get_train_func()`, `get_config()`) | Trainer knows what to run |
| Framework-specific CLI args | **Trainer** (`get_framework_args()`) | Trainer knows its framework's options |
| Scaling & resources | **Trainer** (`num_nodes`, `resources_per_node`) | User sets these on the trainer |
| Serializing function into `command` | **Backend** (`get_command_using_train_func()`) | Backend knows the serialization format |
| Building `TrainJob` CRD / container spec | **Backend** | Backend knows the target platform (K8s, container, local) |
| Injecting distributed args (`rdzv_endpoint`, `nnodes`, etc.) | **Controller** | Controller owns the distributed topology |
| Enforcing policies, quotas, webhooks | **Controller** | Control plane is the final authority |

This separation ensures that:
- Adding a new trainer does **not** require changes to the backend — only a new
  `FuncTrainer` or `ConfigTrainer` subclass.
- Adding a new backend does **not** require changes to trainers — backends consume
  the same `BaseTrainer` interface.
- The controller continues to own distributed coordination args, avoiding conflicts
  between SDK-provided and controller-injected arguments.

### Framework Argument Separation

The current `CustomTrainer.func_args` dict mixes user hyperparameters with framework
arguments. The three-level hierarchy solves this structurally:

| Layer | Method | Contains | Maps to in `TrainJob` CRD |
|---|---|---|---|
| `FuncTrainer` | `get_train_func_args()` | User hyperparameters | Embedded in serialized `trainer.command` |
| `ConfigTrainer` | `get_config()` | Full training configuration | `trainer.args` (parsed by runtime entrypoint) |
| `BaseTrainer` | `get_framework_args()` | Framework CLI args not injected by the controller | Appended to `trainer.args` |

Arguments that the Kubeflow Trainer controller already injects (e.g., `rdzv_endpoint`,
`nnodes`, `nproc_per_node`, `node_rank`) are **excluded** from `get_framework_args()`.
The specialized trainer documentation explicitly lists which arguments it manages vs.
which the controller manages.

### Backend Integration

Each backend (`KubernetesBackend`, `ContainerBackend`, `LocalProcessBackend`) must be
updated to handle `BaseTrainer` instances. The backend reads from the trainer's
interface methods and maps them to platform-specific constructs:

```python
# In KubernetesBackend — building the TrainJob CR:

def _build_trainer_cr(self, runtime, trainer):
    trainer_cr = TrainerV1alpha1Trainer()
    trainer_cr.num_nodes = trainer.num_nodes
    trainer_cr.resources_per_node = trainer.resources_per_node
    trainer_cr.image = trainer.image

    if isinstance(trainer, FuncTrainer):
        # Serialize function into command (same as CustomTrainer today)
        trainer_cr.command = get_command_using_train_func(
            runtime,
            trainer.get_train_func(),
            trainer.get_train_func_args(),
            runtime_config.pip_index_urls if runtime_config else None,
            runtime_config.packages_to_install if runtime_config else None,
        )
        # Framework args go into trainer.args
        framework_args = trainer.get_framework_args()
        if framework_args:
            trainer_cr.args = [
                f"--{k}={v}" for k, v in framework_args.items()
            ]

    elif isinstance(trainer, ConfigTrainer):
        # Config-driven: use runtime's command, pass config as args
        trainer_cr.command = list(runtime.trainer.command)
        trainer_cr.args = [
            f"{k}={v}" for k, v in trainer.get_config().items()
        ]

    return trainer_cr
```

The `runtime_config` parameter is applied uniformly: packages are installed in the
init container, environment variables are set on all training pods.

### Type Hierarchy Diagram

```
                        BaseTrainer (ABC)
                        ├── supported_frameworks (ClassVar)
                        ├── num_nodes, resources_per_node, image
                        ├── get_framework_args()  [abstract]
                        └── validate_runtime()
                              │
              ┌───────────────┴───────────────┐
              │                               │
        FuncTrainer (ABC)              ConfigTrainer (ABC)
        ├── func: Callable             ├── get_config()  [abstract]
        ├── func_args: dict            └── get_framework_args()
        ├── get_train_func()                  │
        └── get_train_func_args()             │  (future, via follow-up proposals)
              │                               │
    ┌─────────┼─────────┬──────────┐    ┌─────┼──────────┬──────────┐
    │         │         │          │    │     │          │          │
  Torch   DeepSpeed   JAX    XGBoost  Torch  Unsloth   VeRL    Axolotl
  Trainer  Trainer  Trainer  Trainer  Tune   Trainer  Trainer  Trainer
              │                      Trainer
              │
    (supports both "deepspeed"
     and "torch" frameworks)


    Existing (unchanged):

    CustomTrainer          BuiltinTrainer         CustomTrainerContainer
    (flat dataclass,       (TorchTuneConfig,      (image-based,
     no base class)         no base class)          no base class)


    New:

    RuntimeConfig
    (per-job env: packages, pip URLs, env vars)
```

---

## User-Facing API Examples

### Before (Current)

```python
from kubeflow.trainer import TrainerClient, CustomTrainer

# User must know the runtime name
client = TrainerClient()

# Must manually look up runtime
runtime = client.get_runtime("torch-distributed")

# Runtime config mixed into trainer
job_id = client.train(
    runtime=runtime,
    trainer=CustomTrainer(
        func=train_pytorch,
        func_args={"lr": 1e-4, "epochs": 10},
        packages_to_install=["transformers", "datasets"],
        pip_index_urls=["https://pypi.org/simple"],
        env={"NCCL_DEBUG": "INFO"},
        num_nodes=4,
        resources_per_node={"gpu": 1, "cpu": 3, "memory": "16Gi"},
    ),
)
```

### After (Proposed)

```python
from kubeflow.trainer import TrainerClient, TorchTrainer, RuntimeConfig

client = TrainerClient()

# Runtime is auto-discovered from trainer.kubeflow.org/framework: torch
# Runtime environment is cleanly separated
job_id = client.train(
    trainer=TorchTrainer(
        func=train_pytorch,
        func_args={"lr": 1e-4, "epochs": 10},
        num_nodes=4,
        resources_per_node={"gpu": 1, "cpu": 3, "memory": "16Gi"},
        max_restarts=3,  # Typed, torch-specific argument
    ),
    runtime_config=RuntimeConfig(
        packages_to_install=["transformers", "datasets"],
        env={"NCCL_DEBUG": "INFO"},
    ),
)
```

**Explicit runtime selection (when multiple runtimes exist for a framework):**

```python
job_id = client.train(
    runtime="torch-elastic",  # Explicit selection
    trainer=TorchTrainer(
        func=train_pytorch,
        func_args={"lr": 1e-4},
        num_nodes=4,
        resources_per_node={"gpu": 2},
    ),
)
```

**DeepSpeed example:**

```python
from kubeflow.trainer import DeepSpeedTrainer, RuntimeConfig

job_id = client.train(
    trainer=DeepSpeedTrainer(
        func=train_deepspeed,
        num_nodes=8,
        resources_per_node={"gpu": 4, "memory": "32Gi"},
        num_proc_per_node=4,
        deepspeed_config={
            "train_batch_size": 32,
            "fp16": {"enabled": True},
            "zero_optimization": {"stage": 2},
        },
    ),
    runtime_config=RuntimeConfig(
        packages_to_install=["deepspeed"],
    ),
)
```

---

## Migration and Backward Compatibility

| Aspect | Impact |
|---|---|
| `CustomTrainer` | **No change.** Remains fully functional. `packages_to_install`, `pip_index_urls`, and `env` fields are retained. |
| `CustomTrainerContainer` | **No change.** |
| `BuiltinTrainer` | **No change in Alpha.** `ConfigTrainer` is designed as its successor (see [BuiltinTrainer Migration Path](#builtintrainer-migration-path)). In Beta, `BuiltinTrainer` will emit a `FutureWarning` directing users to `TorchTuneTrainer`. Formal deprecation occurs at GA. |
| `TrainerClient.train()` | **Additive only.** New `runtime_config` parameter is optional with default `None`. The `trainer` parameter type union is extended to include `BaseTrainer`. |
| `TrainJobTemplate` | **No change in this proposal.** Future work can extend it to support `BaseTrainer` subclasses. |
| `RuntimeConfig` vs `CustomTrainer` fields | When both `RuntimeConfig` and `CustomTrainer` fields are provided, `RuntimeConfig` takes precedence. This is documented but does not break existing code since `RuntimeConfig` defaults to `None`. |
| Python version | No new Python version requirements. Uses `dataclass`, `ABC`, `ClassVar` — all available in Python 3.9+. |
| SDK public exports | New classes are exported from `kubeflow.trainer` (`BaseTrainer`, `FuncTrainer`, `ConfigTrainer`, `TorchTrainer`, `DeepSpeedTrainer`, `JAXTrainer`, `XGBoostTrainer`, `RuntimeConfig`). No existing exports are removed or renamed. |

---

## Test Plan

### Unit Tests

1. **Type hierarchy compliance**: Verify that each `FuncTrainer` subclass correctly
   inherits `func`/`func_args` fields and each `ConfigTrainer` subclass implements
   `get_config()`.
2. **`validate_runtime()` — positive**: Each trainer validates a runtime with a
   matching framework label and compatible `trainer_type`.
3. **`validate_runtime()` — negative framework**: Each trainer raises `ValueError`
   for a runtime with a non-matching framework label.
4. **`validate_runtime()` — negative trainer_type**: `FuncTrainer` subclass raises
   `ValueError` for a `BUILTIN_TRAINER` runtime; `ConfigTrainer` subclass raises
   `ValueError` for a `CUSTOM_TRAINER` runtime.
5. **`get_framework_args()`**: Verify that each trainer returns only non-overlapping
   arguments (excludes controller-injected args).
6. **`RuntimeConfig` defaults**: Verify `None` defaults and precedence over
   `CustomTrainer` fields.
7. **Runtime auto-discovery — single match**: Mock `list_runtimes()` to return one
   matching runtime; verify it is selected.
8. **Runtime auto-discovery — no match**: Mock `list_runtimes()` to return no
   matching runtimes; verify `ValueError`.
9. **Runtime auto-discovery — multiple matches**: Mock `list_runtimes()` to return
   multiple matching runtimes; verify `ValueError` with runtime names in the
   message.

### Integration Tests

1. **End-to-end with `KubernetesBackend`**: Submit a `TorchTrainer` job against a
   cluster with the `torch-distributed` runtime installed; verify the `TrainJob` CR
   is created with the correct runtime reference.
2. **End-to-end with `ContainerBackend`**: Submit a `TorchTrainer` job locally;
   verify the container is launched with the correct entrypoint and arguments.
3. **`RuntimeConfig` application**: Verify that packages from `RuntimeConfig` are
   installed in the training container and env vars are set.

### Backward Compatibility Tests

1. All existing `CustomTrainer` tests pass without modification.
2. All existing `BuiltinTrainer` tests pass without modification.
3. Existing `TrainJobTemplate` usage continues to work.

---

## Implementation Plan

This proposal can be implemented incrementally across multiple PRs:

**Phase 1: Core Type Hierarchy and RuntimeConfig**
- Add `BaseTrainer`, `FuncTrainer`, `ConfigTrainer` to `kubeflow/trainer/types/types.py`
- Add `RuntimeConfig` dataclass
- Add `_resolve_runtime()` to `TrainerClient`
- Extend `TrainerClient.train()` signature
- Unit tests for the type hierarchy and RuntimeConfig

**Phase 2: TorchTrainer**
- Implement `TorchTrainer` (extends `FuncTrainer`)
- Update `KubernetesBackend`, `ContainerBackend`, `LocalProcessBackend` to handle
  `FuncTrainer` and `ConfigTrainer` dispatch
- Integration tests
- Documentation and examples

**Phase 3: DeepSpeedTrainer, JAXTrainer, XGBoostTrainer**
- Implement remaining `FuncTrainer` subclasses
- DeepSpeedTrainer with multi-runtime support (torch and deepspeed/MPI runtimes)
- Framework-specific validation and argument handling
- Tests and documentation

**Phase 4: Public API exports and documentation**
- Export new classes from `kubeflow.trainer.__init__`
- Update SDK documentation on sdk.kubeflow.org
- Add migration guide examples

> **Note:** Phase 1 and Phase 2 should ship together in the same SDK release.
> Releasing the type hierarchy without backend support would make `TorchTrainer`
> importable but unusable, producing a confusing `ValueError` at `train()` time.

---

## Graduation Criteria

### Alpha (target: current cycle)

- `BaseTrainer`, `FuncTrainer`, `ConfigTrainer`, and `RuntimeConfig` types are
  implemented and exported.
- `TorchTrainer` is implemented with full backend support (Kubernetes, Container,
  LocalProcess).
- Runtime auto-discovery and `validate_runtime()` are functional.
- Unit tests cover the type hierarchy, validation (positive and negative), and
  auto-discovery (single-match, no-match, multi-match).
- At least one integration test exercises `TorchTrainer` end-to-end against the
  Kubernetes backend.
- All existing `CustomTrainer` and `BuiltinTrainer` tests continue to pass.
- `RuntimeConfig` merge semantics are implemented and tested.

### Beta

- `DeepSpeedTrainer`, `JAXTrainer`, and `XGBoostTrainer` are implemented.
- At least one `ConfigTrainer` subclass (`TorchTuneTrainer`) is implemented,
  proving the config-driven extension model and beginning `BuiltinTrainer` migration.
- `BuiltinTrainer` emits a `FutureWarning` directing users to `TorchTuneTrainer`.
- SDK documentation on sdk.kubeflow.org covers all new trainer types with examples.
- Migration guide is published.

### GA

- All Tier 1 trainers are stable with no breaking changes for at least one release.
- `BuiltinTrainer` is formally deprecated (removal deferred to a future major version).
- `CustomTrainer` runtime-environment fields (`packages_to_install`, `pip_index_urls`,
  `env`) are formally deprecated in favor of `RuntimeConfig`.
- Community has contributed at least one Tier 2 `ConfigTrainer` subclass.

---

## Open Questions

The following design questions should be resolved before or during implementation:

1. **DeepSpeed launcher detection.** `DeepSpeedTrainer` supports both `torchrun` and
   `mpirun` launchers. How should `validate_runtime()` detect which launcher a runtime
   uses? The `Runtime` type currently does not expose launcher metadata. Options:
   (a) inspect `runtime.trainer.command`, (b) add a launcher label to runtimes,
   (c) defer validation to the controller.

2. **`ConfigTrainer.get_config()` return type.** The current return type `dict` is
   untyped. Should `ConfigTrainer` subclasses return a typed configuration object
   instead (e.g., a Pydantic model or typed dataclass) to get the same static-analysis
   benefits that `FuncTrainer` provides via typed fields?

3. **Observability.** When runtime auto-discovery selects a runtime, should the SDK
   log the selected runtime name and framework at `INFO` level? This would aid
   debugging but adds a logging dependency.

4. **`resources_per_node` typing.** The current `Optional[dict]` is untyped. Should
   this be a structured type (e.g., `ResourceRequirements` dataclass with `cpu`,
   `memory`, `gpu` fields) to enable IDE autocomplete and validation?

5. **Backend validation safety net.** Currently, `validate_runtime()` is only called
   via `TrainerClient._resolve_runtime()`. Should backends also call
   `validate_runtime()` as a defensive check, in case trainers are passed to backends
   directly?

---

## Alternatives Considered

### 1. Extend CustomTrainer with a `framework` field instead of new classes

Add a `framework: Optional[str]` field to `CustomTrainer` and use it for runtime
discovery and validation.

**Rejected because:**
- Does not provide a place for framework-specific typed arguments (`max_restarts`,
  `num_proc_per_node`).
- Does not enable the Tier 2 extension model.
- Violates the open-closed principle: the `CustomTrainer` class would need to grow
  with each new framework.

### 2. Use Pydantic `BaseModel` instead of `@dataclass`

Use Pydantic for automatic validation, serialization, and schema generation.

**Rejected because:**
- The existing SDK codebase uses `@dataclass` exclusively. Introducing Pydantic
  would add a dependency and create an inconsistency in the codebase.
- Pydantic validation can be replicated with `__post_init__` where needed.

### 3. Put RuntimeConfig inside BaseTrainer instead of as a separate parameter

Make `RuntimeConfig` a field on `BaseTrainer` so that each trainer carries its own
runtime config.

**Rejected because:**
- Runtime configuration (packages, env vars) is orthogonal to trainer type. The
  same `RuntimeConfig` should be usable with `CustomTrainer`,
  `CustomTrainerContainer`, or any `BaseTrainer` subclass.
- Keeping it as a separate `train()` parameter maintains clean separation of concerns.
- Users may want custom env for initializers too. A separate `train()` parameter
  can be extended to apply to both trainers and initializers without changing trainer
  classes.

### 4. Automatic runtime selection with scoring/ranking instead of strict single-match

When multiple runtimes match a framework, automatically pick the "best" one using a
scoring heuristic (e.g., prefer non-deprecated, prefer more specific labels).

**Rejected because:**
- Implicit selection heuristics are fragile and hard to debug. When multiple runtimes
  exist for the same framework, it is a deliberate platform configuration and the
  user should explicitly choose.
- A clear error message listing available runtimes is more useful than a possibly
  wrong automatic selection.

### 5. Flat hierarchy: all trainers inherit directly from BaseTrainer

Have all concrete trainers (both function-driven and config-driven) inherit directly
from `BaseTrainer` without the `FuncTrainer` / `ConfigTrainer` intermediate layer.

**Rejected because:**
- Config-driven trainers would carry `get_train_func()` returning `None` — semantically
  incorrect and error-prone.
- Function-driven trainers would each redeclare `func` and `func_args` fields —
  unnecessary repetition.
- Backend dispatch would rely on runtime checks (`if trainer.get_train_func() is None`)
  instead of type checks (`isinstance(trainer, ConfigTrainer)`).
- The intermediate classes encode the fundamental difference between the two trainer
  modes at the type level, making the API self-documenting.

### 6. Have specialized trainers inherit from CustomTrainer

Make `TorchTrainer` a subclass of `CustomTrainer` instead of a new `BaseTrainer`
hierarchy.

**Rejected because:**
- `CustomTrainer` carries runtime-environment fields (`packages_to_install`,
  `pip_index_urls`, `env`) that specialized trainers should not expose (those belong
  in `RuntimeConfig`).
- Inheriting from `CustomTrainer` would force specialized trainers to carry fields
  that violate the separation of concerns this proposal aims to achieve.

---

## References

- [KEP-2170: Kubeflow Trainer V2 API](https://github.com/kubeflow/trainer/blob/master/docs/proposals/2170-kubeflow-trainer-v2/README.md)
- [Kubeflow SDK Repository](https://github.com/kubeflow/sdk)
- [Kubeflow Trainer Repository](https://github.com/kubeflow/trainer)
- [Kubeflow Community Proposal Workflow](https://github.com/kubeflow/community/blob/master/proposal-workflow.md)
- [Runtime Guide — trainer.kubeflow.org/framework label](https://www.kubeflow.org/docs/components/trainer/operator-guides/runtime/)
- [Kubeflow Trainer Getting Started](https://www.kubeflow.org/docs/components/trainer/getting-started/)
- [SDK Types Source Code](https://github.com/kubeflow/sdk/blob/main/kubeflow/trainer/types/types.py)
- [SDK TrainerClient Source Code](https://github.com/kubeflow/sdk/blob/main/kubeflow/trainer/api/trainer_client.py)
