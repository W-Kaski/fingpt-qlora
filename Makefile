.PHONY: data manifest demo clean install validate

install:
	pip install -r requirements.txt

validate:
	python -m unittest discover tests
	python -m compileall -q src tests

data:
	python -m src.data.download
	python -m src.data.preprocess
	python -m src.data.format_chat
	python -m src.data.merge_datasets
	python -m src.data.splits

manifest:
	python -m src.data.manifest --splits-dir data/splits --output results/data_manifest_v3.json

demo:
	python demo/app.py

clean:
	rm -rf data/raw/* data/processed/* data/splits/*
	rm -rf outputs/* results/training_logs/* results/eval_outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
