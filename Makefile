.PHONY: install test run-dashboard run-lakehouse-bootstrap tree

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

run-dashboard:
	python -m streamlit run src/socialpulse_v2/dashboard/app.py

run-lakehouse-bootstrap:
	python -m socialpulse_v2.orchestration.bootstrap_lakehouse

tree:
	find . -maxdepth 4 | sort