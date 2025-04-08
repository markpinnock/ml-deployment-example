SHELL := /bin/bash

# Variables
include .env
IMAGE_NAME = "ml-deployment"
COMMIT_HASH = $$(git rev-parse --short=8 HEAD)


# Clean up
.PHONY: clean
clean_cache:
	rm -rf .mypy_cache \
	rm -rf .pytest_cache \
	rm -rf .ruff_cache


# Compile pip requirements from uv
.PHONY: pip_compile
pip_compile:
	uv pip compile pyproject.toml -o requirements.txt


# Build docker image
.PHONY: build_docker
build_docker: pip_compile
	docker build -t $(IMAGE_NAME):$(COMMIT_HASH) -t $(IMAGE_NAME):latest .


# Run docker image locally
.PHONY: run_docker
run_docker:
	docker run -d -p 80:80 ml-deployment:latest
