SHELL := /bin/bash

# Variables
include .env
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
# Args:
# - IMAGE_NAME: name of the image to build
.PHONY: build
build:
	docker build -t $(DOCKER_USERNAME)/$(IMAGE_NAME):$(COMMIT_HASH) \
	-t $(DOCKER_USERNAME)/$(IMAGE_NAME):latest $(IMAGE_NAME)


# Run docker image locally
# Args:
# - IMAGE_NAME: name of the image to build
.PHONY: run
run:
	docker run -d -p 80:80 $(IMAGE_NAME):latest


# Build docker image and push to DockerHub
# Args:
# - IMAGE_NAME: name of the image to build
.PHONY: build_and_push
build_and_push: build
	docker login -u $(DOCKER_USERNAME) -p $(DOCKER_PASSWORD)
	docker push $(DOCKER_USERNAME)/$(IMAGE_NAME):$(COMMIT_HASH)
	docker push $(DOCKER_USERNAME)/$(IMAGE_NAME):latest
