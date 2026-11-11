PYTHON ?= python3
VENV ?= .venv-512
RUN_DIR ?= data/reproduce-512
SEED ?= 20261121

.PHONY: reproduce-512 setup-512
setup-512:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 torchvision==0.20.1
	$(VENV)/bin/python -m pip install -r requirements-reproduce-512.lock

reproduce-512: setup-512
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(VENV)/bin/python scripts/reproduce_512.py --output $(RUN_DIR) --seed $(SEED) $(EXTRA_ARGS)
