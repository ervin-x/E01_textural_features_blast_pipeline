SHELL := /bin/zsh
PYTHON := ./.venv/bin/python
DATA_ROOT ?=
MASKS_ROOT ?=
PATHS_CONFIG ?=

RUN_ARGS :=

ifneq ($(strip $(DATA_ROOT)),)
RUN_ARGS += --data-root "$(DATA_ROOT)"
endif

ifneq ($(strip $(MASKS_ROOT)),)
RUN_ARGS += --masks-root "$(MASKS_ROOT)"
endif

ifneq ($(strip $(PATHS_CONFIG)),)
RUN_ARGS += --paths-config "$(PATHS_CONFIG)"
endif

.PHONY: foundation h1 h2 h3 h4 h5 all test repro-check clean-pyc

foundation:
	$(PYTHON) scripts/run_foundation.py $(RUN_ARGS)

h1:
	$(PYTHON) scripts/run_h1_interpretable_features.py

h2:
	$(PYTHON) scripts/run_h2_mask_effect.py

h3:
	$(PYTHON) scripts/run_h3_noise_regimes.py

h4:
	$(PYTHON) scripts/run_h4_aggregation.py

h5:
	$(PYTHON) scripts/run_h5_deep_vs_handcrafted.py

all:
	$(PYTHON) scripts/run_all_hypotheses.py $(RUN_ARGS)

test:
	$(PYTHON) -m unittest discover -s tests -v

repro-check: all test

clean-pyc:
	find scripts src tests -type d -name '__pycache__' -prune -exec rm -rf {} +
