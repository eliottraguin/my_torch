##
## EPITECH PROJECT, 2026
## torch
## File description:
## Makefile
##

TEST_ENV	=	.venv-tests
TEST_PYTHON	=	$(TEST_ENV)/bin/python

DASHBOARD_ENV = .venv-dashboard
DASHBOARD_PYTHON = $(DASHBOARD_ENV)/bin/python
DASHBOARD_PORT ?= 8501

NAME1	=	my_torch_analyzer
NAME2	=	my_torch_generator

all:
	python3 -m zipapp src -m analyzer:main -o $(NAME1) -p "/usr/bin/env python3"
	python3 -m zipapp src -m generator:main -o $(NAME2) -p "/usr/bin/env python3"

re:	fclean all

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

fclean:	clean
	$(RM) $(NAME1) $(NAME2) .coverage .coverage.*
	$(RM) -r htmlcov

$(TEST_ENV)/.dependencies-installed: tests/requirements-dev.txt
	python3 -m venv $(TEST_ENV)
	$(TEST_PYTHON) -m pip install -r tests/requirements-dev.txt
	touch $@

tests_run: $(TEST_ENV)/.dependencies-installed
	PYTHONPATH="$(CURDIR)/src$${PYTHONPATH:+:$$PYTHONPATH}" $(TEST_PYTHON) -m coverage run -m unittest discover -s tests -t . -v
	$(TEST_PYTHON) -m coverage report
	$(TEST_PYTHON) -m coverage html

$(DASHBOARD_ENV)/.dependencies-installed: bonus/dashboard/requirements.txt scripts/requirements-dataset.txt
	python3 -m venv $(DASHBOARD_ENV)
	$(DASHBOARD_PYTHON) -m pip install -r bonus/dashboard/requirements.txt
	touch $@

dashboard: $(DASHBOARD_ENV)/.dependencies-installed
	cd bonus/dashboard && $(abspath $(DASHBOARD_PYTHON)) -m streamlit run app.py --server.port $(DASHBOARD_PORT)

dashboard_tests: $(DASHBOARD_ENV)/.dependencies-installed
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$(CURDIR):$(CURDIR)/src" $(DASHBOARD_PYTHON) -m unittest discover -s bonus/dashboard/tests -v

.PHONY:	all re clean fclean tests_run dashboard dashboard_tests
