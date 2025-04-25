# Makefile

py-files := $(shell find . -name '*.py')

format:
	@black $(py-files)
	@ruff format $(py-files)
	@ruff check --fix $(py-files)
.PHONY: format

static-checks:
	@black --diff --check $(py-files)
	@ruff check $(py-files)
	@mypy --install-types --non-interactive $(py-files)
.PHONY: lint
