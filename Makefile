#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ccml_lightning
PYTHON_VERSION = 3.10.16
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################



## Install Python Dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt




## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using flake8 and black (use `make format` to do formatting)
.PHONY: lint
lint:
	flake8 ml
	isort --check --diff --profile black ml
	black --check --config pyproject.toml ml


## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml ml



## Set up python interpreter environment
.PHONY: create_environment
create_environment:

	conda create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y

	@echo ">>> conda env created. Activate with:\nconda activate $(PROJECT_NAME)"




## Activate python environment
.PHONY: activate_environment
activate_environment:
	conda activate $(PROJECT_NAME)



.PHONY: env
env: create_environment activate_environment requirements


.PHONY: setup_hooks
setup_hooks:
	@if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then \
		pre-commit install; \
		pre-commit autoupdate --repo https://github.com/pre-commit/pre-commit-hooks; \
		nbautoexport install; \
		nbautoexport configure notebooks --overwrite; \
	else \
		echo "Not inside a Git repository. Skipping pre-commit setup."; \
	fi


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Make Dataset
.PHONY: data
data:
	$(PYTHON_INTERPRETER) ml/dataset.py


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
