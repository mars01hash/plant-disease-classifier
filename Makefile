.PHONY: help install train api test docker-build docker-run clean lint format

help:
	@echo "Plant Disease Classifier - Common Tasks"
	@echo "========================================"
	@echo "make install        - Install dependencies"
	@echo "make data          - Generate sample data"
	@echo "make train         - Train the model"
	@echo "make api           - Start API server"
	@echo "make test          - Run tests"
	@echo "make lint          - Check code quality"
	@echo "make format        - Format code with black"
	@echo "make docker-build  - Build Docker image"
	@echo "make docker-run    - Run Docker container"
	@echo "make clean         - Remove generated files"
	@echo ""

install:
	pip install -r requirements.txt
	@echo "Dependencies installed!"

data:
	python scripts/create_sample_data.py --num-samples 50
	@echo "Sample data created!"

train:
	python src/train.py

api:
	python src/api.py

test:
	pytest tests/ -v --tb=short

lint:
	flake8 src/ tests/ --max-line-length=120
	@echo "Linting complete!"

format:
	black src/ tests/ --line-length=120
	@echo "Code formatted!"

docker-build:
	docker build -t plant-classifier:latest .
	@echo "Docker image built!"

docker-run:
	docker run -p 8000:8000 plant-classifier:latest

docker-compose-up:
	docker-compose up

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	@echo "Cleaned!"

setup-dev: install
	pip install pytest pytest-cov black flake8
	@echo "Development environment setup complete!"

.DEFAULT_GOAL := help
