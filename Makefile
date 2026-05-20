.PHONY: env install smoke test lint fmt vendor \
        prep-qm9 prep-crossdocked prep-plinder \
        train-qm9-edm train-qm9-fm train-sbdd-edm train-sbdd-fm \
        eval nightly-boltz clean

CONDA ?= mamba
ENV_NAME ?= genmol
# Optional: install env at an absolute prefix instead of the default conda envs dir.
# Example: `make env ENV_PREFIX=/project2/katritch_223/aoxu/envs/genmol`
ENV_PREFIX ?=
PY ?= python

ifeq ($(strip $(ENV_PREFIX)),)
ENV_TARGET := -n $(ENV_NAME)
else
ENV_TARGET := --prefix $(ENV_PREFIX)
endif

# ============ Environment ============

env:
	$(CONDA) env create $(ENV_TARGET) -f environment.yml \
	  || $(CONDA) env update $(ENV_TARGET) -f environment.yml

install:
	pip install -e ".[dev]"
	pre-commit install || true

vendor:
	bash tools/vendor_egnn.sh

# ============ Quality gates ============

smoke:
	pytest tests/smoke/ -x -v

test:
	pytest tests/ -x

lint:
	ruff check genmol/ tests/ scripts/
	black --check genmol/ tests/ scripts/

fmt:
	ruff check --fix genmol/ tests/ scripts/
	black genmol/ tests/ scripts/
	isort genmol/ tests/ scripts/

# ============ Data prep ============

prep-qm9:
	$(PY) scripts/prepare_qm9.py

prep-crossdocked:
	$(PY) scripts/prepare_crossdocked.py

prep-plinder:
	$(PY) tools/plinder/stage_from_mount.py

# ============ Training (M1–M4) ============

train-qm9-edm:
	$(PY) -m scripts.train experiment=qm9_edm

train-qm9-fm:
	$(PY) -m scripts.train experiment=qm9_flow

train-sbdd-edm:
	$(PY) -m scripts.train experiment=crossdocked_edm

train-sbdd-fm:
	$(PY) -m scripts.train experiment=crossdocked_flow

# ============ Eval ============

eval:
	@test -n "$(CKPT)" || (echo "Usage: make eval CKPT=path/to/model.ckpt"; exit 1)
	$(PY) scripts/evaluate.py ckpt=$(CKPT)

nightly-boltz:
	$(PY) scripts/nightly_boltz.py

# ============ Cleanup ============

clean:
	rm -rf logs/ runs/ checkpoints/ outputs/ multirun/ wandb/ lightning_logs/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
