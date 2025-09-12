.PHONY: install build format lint test clean all

install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

build:
	python -m src.app.cli

format:
	black src tests scripts

lint:
	flake8 src tests scripts

test:
	pytest -vv --cov=src --cov-report=term-missing tests

clean:
	rm -rf __pycache__ .pytest_cache .coverage htmlcov

all: install format lint test
