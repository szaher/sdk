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


# Setting SHELL to bash allows bash commands to be executed by recipes.
# Options are set to exit when a recipe line exits non-zero or a piped command fails.
SHELL = /usr/bin/env bash -o pipefail
.SHELLFLAGS := -eu -o pipefail -c

# run the whole recipe in one shell (so one cd covers all lines)
.ONESHELL:
.PHONY: verify


PROJECT_DIR := $(shell dirname $(abspath $(lastword $(MAKEFILE_LIST))))

PY_DIR := $(PROJECT_DIR)/python

REPO := github.com/kubeflow/sdk
# Location to install tool binaries
LOCALBIN ?= $(PROJECT_DIR)/bin

# Tool versions
UV_VERSION=0.7.2
RUFF_VERSION=0.12.3


# Tool binaries
UV ?= $(which uv)
RUFF ?= $(which ruff)

##@ General

# The help target prints out all targets with their descriptions organized
# beneath their categories. The categories are represented by '##@' and the
# target descriptions by '##'. The awk commands is responsible for reading the
# entire set of makefiles included in this invocation, looking for lines of the
# file as xyz: ## something, and then pretty-format the target and help. Then,
# if there's a line with ##@ something, that gets pretty-printed as a category.
# More info on the usage of ANSI control characters for terminal formatting:
# https://en.wikipedia.org/wiki/ANSI_escape_code#SGR_parameters
# More info on the awk command:
# http://linuxcommand.org/lc3_adv_awk.php

help: ## Display this help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Development

# Instructions to download tools for development.

.PHONY: install-uv
install-uv: ## Install UV tool on local system
	curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh


.PHONY: install-ruff
install-ruff: ## Install UV tool on local system
	uv tool install "ruff@$(RUFF_VERSION)"

# run checks
.PHONY: verify-lock
verify-lock: ## Check UV lockfile
	@cd $(PY_DIR) && uv lock --check

.PHONY: verify-ruff
verify-ruff: ## Run ruff check
	@cd $(PY_DIR) && uvx ruff check --show-fixes

.PHONY: verify
verify: verify-lock verify-ruff  ## Verify everything is Ok for python sdk



.PHONY: test-python
test-python: ## Run Python unit test.
	uv install --dev
	PYTHONPATH=$(PROJECT_DIR) pytest ./python/kubeflow

