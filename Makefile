.PHONY: install test run-dashboard run-lakehouse-bootstrap run-historical-bootstrap run-daily-plan run-daily-youtube run-bronze-daily-ingestion kafka-up kafka-down kafka-logs run-kafka-producer run-kafka-consumer run-silver-youtube-comments run-silver-youtube-sentiment run-gold-youtube-sentiment run-gold-daily-overview run-gold-youtube-sentiment-descriptive run-predictive test-gold tree run-gold-youtube-sentiment-all

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

run-dashboard:
	PYTHONPATH=src python -m streamlit run src/socialpulse_v2/dashboard/app.py

run-lakehouse-bootstrap:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.bootstrap_lakehouse

run-historical-bootstrap:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.bootstrap_historical_youtube

run-daily-plan:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.plan_daily_collection

run-daily-youtube:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_daily_youtube_collection

run-bronze-daily-ingestion:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_bronze_daily_ingestion

kafka-up:
	docker compose -f docker-compose.kafka.yml up -d

kafka-down:
	docker compose -f docker-compose.kafka.yml down

kafka-logs:
	docker compose -f docker-compose.kafka.yml logs -f kafka

run-kafka-producer:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.publish_youtube_comments_to_kafka

run-kafka-consumer:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.consume_youtube_comments_to_bronze

run-silver-youtube-comments:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_silver_youtube_comments

run-silver-youtube-sentiment:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_silver_youtube_comments_sentiment

run-gold-youtube-sentiment:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_gold_youtube_sentiment

run-gold-daily-overview:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_gold_daily_overview

run-gold-youtube-sentiment-descriptive:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_gold_youtube_sentiment_descriptive

run-predictive:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_gold_youtube_comments_predictive

test-gold:
	pytest -q tests/gold

run-gold-youtube-sentiment-all:
	$(MAKE) run-gold-youtube-sentiment
	$(MAKE) run-gold-youtube-sentiment-descriptive

run-custom-youtube-query:
	PYTHONPATH=src python -m socialpulse_v2.orchestration.run_custom_youtube_query_collection

tree:
	find . -maxdepth 4 | sort