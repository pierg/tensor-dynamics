# Docker variables
IMAGE_NAME=neural_networks
VERSION=latest

.PHONY: clean_all format lint clean_results install run tb stoptb docker-push

clean_all:
	# Remove Python compiled files and cache directories
	find . -type f -name "*.pyc" -exec rm -f {} +
	find . -type d -name "__pycache__" -exec rm -rf {} +
	
	# Remove virtual environment directory
	rm -rf .venv

	# Remove build directories
	rm -rf build dist *.egg-info
	
	# Remove poetry.lock file to ensure fresh dependencies
	rm -f poetry.lock

format:
	# Format code using Black
	poetry run black .

lint:
	# Lint project using flake8 and mypy for static type checking
	poetry run flake8 .
	poetry run mypy .

clean_results:
	# Remove content of results directory besides "saved" directory
	@echo "Cleaning results directory..."
	@find results/ -mindepth 1 -maxdepth 1 ! -name 'saved' -exec rm -rf {} +
	@echo "Done cleaning results directory."


install:
	# Install dependencies with pip from the requirements.txt file
	pip install -r requirements.txt

.PHONY: run tb

# Launch TensorBoard if not already running
tb:
	@if ! pgrep -x "tensorboard" > /dev/null; then \
		echo "Starting TensorBoard..."; \
		tensorboard --logdir ./logs & \
	else \
		echo "TensorBoard is already running."; \
	fi

# The run command requires one mandatory argument (data directory) and an optional list of configurations.
# For example: make run DATA_DIR=~/Documents/data CONFIGS='config_Alex config_P'
run: tb
ifdef DATA_DIR
	. .venv/bin/activate && python src/main.py --data_dir=$(DATA_DIR) $(CONFIGS)
else
	@echo "Error: DATA_DIR is undefined. Usage: make run DATA_DIR=<data-directory> [CONFIGS='<config1> <config2> ...']"
endif


# Terminate the TensorBoard process
stoptb:
	@echo "Attempting to stop TensorBoard..."
	@pkill -f "tensorboard" || echo "Could not find a running TensorBoard process."


# Load secrets
include .secrets
export $(shell sed 's/=.*//' .secrets)

# Build and push to Docker Hub
docker-push:
	# Build the image
	docker build -t $(DOCKER_USER)/$(IMAGE_NAME):$(VERSION) .

	# Log in to Docker Hub
	@echo "$(DOCKER_PASS)" | docker login --username $(DOCKER_USER) --password-stdin

	# Push the image
	docker push $(DOCKER_USER)/$(IMAGE_NAME):$(VERSION)

	# Optional: Log out from Docker Hub
	docker logout

