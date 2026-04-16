.PHONY: install test run-dashboard tree

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

run-dashboard:
	python -m streamlit run src/socialpulse_v2/dashboard/app.py

tree:
	find . -maxdepth 3 | sort