.PHONY: data demo clean install

install:
	pip install -r demo/requirements.txt

data:
	python -m src.data.download
	python -m src.data.preprocess
	python -m src.data.format_chat
	python -m src.data.merge_datasets
	python -m src.data.splits

demo:
	python demo/app.py

clean:
	rm -rf data/raw/* data/processed/* data/splits/*
	rm -rf outputs/* results/training_logs/* results/eval_outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
