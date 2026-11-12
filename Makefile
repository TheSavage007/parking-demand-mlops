.PHONY: install lint test train serve docker

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

test:
	pytest -q

train:
	PYTHONPATH=src python -m parking_mlops.train

serve:
	PYTHONPATH=src uvicorn app.main:app --reload

docker:
	docker build -t parking-demand-api .
	docker run --rm -p 8000:8000 parking-demand-api
