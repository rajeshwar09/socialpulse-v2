.PHONY: install test run-dashboard run-lakehouse-bootstrap run-historical-bootstrap run-daily-plan tree

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

run-dashboard:
	python -m streamlit run src/socialpulse_v2/dashboard/app.py

run-lakehouse-bootstrap:
	python -m socialpulse_v2.orchestration.bootstrap_lakehouse

run-historical-bootstrap:
	python -m socialpulse_v2.orchestration.bootstrap_historical_youtube

run-daily-plan:
	python -m socialpulse_v2.orchestration.plan_daily_collection

tree:
	find . -maxdepth 4 | sort