# Variables for Docker configuration
IMAGE_NAME=neural_networks
VERSION=latest

# Phony targets for make
.PHONY: clean_all format lint mypy clean_results install run tb stoptb docker-buildx docker-push

# Clean up compiled Python files, cache directories, virtual environment, build directories, and lock file
clean_all:
	find . -type f -name "*.pyc" -exec rm -f {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .venv
	rm -rf build dist *.egg-info
	rm -f poetry.lock

# Format code using Black and sort imports using isort
format:
	poetry run black .
	poetry run isort .

# Lint code using autoflake and pylint
lint:
	poetry run autoflake .
	poetry run pylint src/

# Static type checking using mypy
mypy:
	poetry run mypy .

# Clean the results directory except the 'saved' folder
clean_results:
	@echo "Cleaning results directory..."
	@find results/ -mindepth 1 -maxdepth 1 ! -name 'saved' -exec rm -rf {} +
	@echo "Done cleaning results directory."

# Install dependencies from requirements.txt
install:
	pip install -r requirements.txt

# Launch TensorBoard if not already running
tb:
	@if ! pgrep -x "tensorboard" > /dev/null; then \
		echo "Starting TensorBoard..."; \
		tensorboard --logdir ./logs & \
	else \
		echo "TensorBoard is already running."; \
	fi

# Run command with optional configurations
run: tb
ifdef DATA_DIR
	. .venv/bin/activate && python src/main.py $(CONFIGS)
else
	@echo "Error: DATA_DIR is undefined. Usage: make run CONFIGS='<config1> <config2> ...'"
endif

# Stop TensorBoard process
stoptb:
	@echo "Attempting to stop TensorBoard..."
	@pkill -f "tensorboard" || echo "Could not find a running TensorBoard process."

# Include .secrets file
include .secrets
export $(shell sed 's/=.*//' .secrets)

# Docker buildx for multi-architecture container images
docker-buildx:
	BUILDER_NAME := multi-arch-builder
	BUILDER_EXISTS := $(shell docker buildx inspect $(BUILDER_NAME) >/dev/null && docker buildx inspect $(BUILDER_NAME) --bootstrap)
	ifeq ($(BUILDER_EXISTS),)
		docker buildx create --name $(BUILDER_NAME) --use
	endif

# Push images to Docker Hub with multi-architecture support
docker-push:
	@echo "$(DOCKER_PASS)" | docker login --username $(DOCKER_USER) --password-stdin
	docker buildx inspect --bootstrap
	docker buildx build --platform linux/amd64,linux/arm64 -t $(DOCKER_USER)/$(IMAGE_NAME):$(VERSION) --push .
	docker logout
