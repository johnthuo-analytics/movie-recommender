.PHONY: install test lint evaluate train serve docker
install:
	pip install -r requirements.txt ruff
test:
	pytest -q
lint:
	ruff check . && ruff format --check .
evaluate:
	python evaluate.py
train:
	python train.py
serve:
	uvicorn api:app --reload
docker: train
	docker build -t movie-recommender .
	docker run -p 8000:8000 movie-recommender
