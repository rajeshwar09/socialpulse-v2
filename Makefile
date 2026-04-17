.PHONY: install test run-dashboard run-lakehouse-bootstrap run-historical-bootstrap run-daily-plan run-daily-youtube tree

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

run-daily-youtube:
	python -m socialpulse_v2.orchestration.run_daily_youtube_collection

run-bronze-daily-ingestion:
	python -m socialpulse_v2.orchestration.run_bronze_daily_ingestion

kafka-up:
	docker compose -f docker-compose.kafka.yml up -d

kafka-down:
	docker compose -f docker-compose.kafka.yml down

kafka-logs:
	docker compose -f docker-compose.kafka.yml logs -f kafka

run-kafka-producer:
	python -m socialpulse_v2.orchestration.publish_youtube_comments_to_kafka

run-kafka-consumer:
	python -m socialpulse_v2.orchestration.consume_youtube_comments_to_bronze

tree:
	find . -maxdepth 4 | sort
run-silver-youtube-comments:
	python -m socialpulse_v2.orchestration.run_silver_youtube_comments

