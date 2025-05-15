# Makefile

py-files := $(shell find . -name '*.py' -not -path "*/run_*/*" -not -path "*/build/*")

install:
	@pip install --upgrade --upgrade-strategy eager -r requirements.txt
.PHONY: install

install-dev:
	@pip install ruff mypy
.PHONY: install-dev

format:
	@ruff format $(py-files)
	@ruff check --fix $(py-files)
.PHONY: format

static-checks:
	@mkdir -p .mypy_cache
	@ruff check $(py-files)
	@mypy --install-types --non-interactive $(py-files)
.PHONY: lint

notebook:
	jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser
.PHONY: notebook
