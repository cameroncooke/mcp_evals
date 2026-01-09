PYTHON = $(shell [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3)

CONFIG ?= config.yaml
TASKS ?= tasks.yaml
TRIALS ?= 10
RUN ?=
ARGS ?=

.PHONY: help venv deps deps-dev test typecheck check run resume rebuild validate validate_only smoke estimate

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' Makefile | awk 'BEGIN {FS=":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

venv: ## Create virtualenv in .venv
	python3 -m venv .venv
	.venv/bin/python -m pip install -U pip

deps: venv ## Install Python deps into .venv
	.venv/bin/python -m pip install -r requirements.txt

deps-dev: deps ## Install dev deps (typecheck)
	.venv/bin/python -m pip install -r requirements-dev.txt

test: ## Run unit tests
	$(PYTHON) -m unittest discover -s tests

typecheck: ## Run Pyright type checking
	.venv/bin/pyright

check: test typecheck ## Run tests + typecheck

run: ## Run eval suite (CONFIG/TASKS/TRIALS/ARGS supported)
	$(PYTHON) run_suite.py --config $(CONFIG) --tasks $(TASKS) --trials $(TRIALS) $(ARGS)

rebuild: ## Rebuild reports (RUN required, ARGS supported)
	@test -n "$(RUN)" || (echo "RUN is required, e.g. RUN=runs/20260119_114416"; exit 1)
	$(PYTHON) rebuild_reports.py --run $(RUN) --config $(CONFIG) --in-place $(ARGS)

validate: ## Validate reference solutions (runs trials afterward)
	$(PYTHON) run_suite.py --config $(CONFIG) --tasks $(TASKS) --validate-tasks $(ARGS)

validate_only: ## Validate reference solutions and exit
	$(PYTHON) run_suite.py --config $(CONFIG) --tasks $(TASKS) --validate-tasks-only $(ARGS)

smoke: ## Run smoke tasks with trials=1
	$(PYTHON) run_suite.py --config $(CONFIG) --tasks tasks.yaml --task-ids smoke_build_install_launch --trials 1 $(ARGS)

resume: ## Resume an interrupted run (RUN required)
	@test -n "$(RUN)" || (echo "RUN is required, e.g. RUN=runs/20260119_114416"; exit 1)
	$(PYTHON) run_suite.py --config $(CONFIG) --tasks $(TASKS) --trials $(TRIALS) --resume $(RUN) $(ARGS)

estimate: ## Estimate cost/time for a run (RUN required, TRIALS optional)
	@test -n "$(RUN)" || (echo "RUN is required, e.g. RUN=runs/20260119_114416"; exit 1)
	$(PYTHON) estimate_run.py --run $(RUN) --trials $(TRIALS)
