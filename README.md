## SocialPulse Lakehouse Project

SocialPulse is a Big Data Analytics project for social-listening and sentiment analysis on YouTube comments. The project collects social media comments, stores them in a lakehouse-style data layout, processes them using PySpark and Delta Lake, builds analytics marts, and displays the final KPIs and insights in a Streamlit dashboard.

> Current evaluation scope: YouTube pipeline is the main working scope. Reddit support can be kept as a future extension if API access or dataset availability is added later.

---

##  Project Features

- YouTube comment collection using YouTube Data API.
- Bronze, Silver, and Gold lakehouse layers.
- PySpark + Delta Lake based transformation pipeline.
- Sentiment analysis and social listening metrics.
- Topic-wise and time-wise analytics marts.
- Predictive and prescriptive insight tables.
- MongoDB connection for storing and reading collected data as the operational database.
- Streamlit dashboard for final demonstration.
- Modular project structure so that Reddit or other sources can be added later.

---


##  System Requirements

Recommended system:

- macOS / Linux / Ubuntu VM
- Python 3.11+ or Python 3.13 if your project is already configured with it
- Java 17 or Java 11
- Git
- Internet connection for YouTube API collection
- YouTube Data API key
- MongoDB running locally, through Docker, or through MongoDB Atlas

Check installed versions:

```bash
python3 --version
java -version
git --version
```

---

##  Create and Activate Python Virtual Environment

Create virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Verify important packages:

```bash
python -c "import pyspark; print('PySpark OK')"
python -c "import delta; print('Delta Lake OK')"
python -c "import streamlit; print('Streamlit OK')"
```

---

##  Environment Configuration

Create `.env` file from example:

```bash
cp .env.example .env
```

Open `.env`:

```bash
nano .env
```

Add or update the following values:

```env
SOCIALPULSE_APP_NAME=SocialPulse V2
SOCIALPULSE_ENV=dev
SOCIALPULSE_TIMEZONE=Asia/Kolkata
SOCIALPULSE_DATA_ROOT=./data
SOCIALPULSE_LOG_LEVEL=INFO
SOCIALPULSE_DASHBOARD_THEME=dark

YOUTUBE_API_KEY=YOUR_YOUTUBE_API_KEY_HERE

MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=socialpulse
MONGO_YOUTUBE_COLLECTION=youtube_comments
MONGO_VIDEO_COLLECTION=youtube_videos
MONGO_RUN_COLLECTION=collection_runs

SPARK_APP_NAME=SocialPulseLakehouse
SPARK_MASTER=local[*]
SPARK_SQL_SHUFFLE_PARTITIONS=4
SPARK_WAREHOUSE_DIR=./data/spark-warehouse
```

Save the file.

For `nano`:

```text
CTRL + O, Enter, CTRL + X
```

---

##  Create Required Data Folders

Run:

```bash
mkdir -p data/raw/youtube
mkdir -p data/bronze/youtube
mkdir -p data/silver/youtube
mkdir -p data/gold/youtube
mkdir -p data/checkpoints
mkdir -p data/spark-warehouse
mkdir -p logs
```

---

##  Check Spark and Delta Lake Setup

Run the development check script if available:

```bash
python -m scripts.dev.check_spark_delta
```

If your project uses shell scripts:

```bash
bash scripts/dev/check_spark_delta.sh
```

Expected result:

```text
Spark session created successfully
Delta Lake read/write test successful
```

A warning like this is usually acceptable on local machines:

```text
WARN NativeCodeLoader: Unable to load native-hadoop library for your platform
```

---

##  Run YouTube Data Collection

Small demo collection:

```bash
bash ./scripts/run/run_youtube_collection.sh \
  --run-label demo_run \
  --max-keywords 3 \
  --max-videos-per-keyword 5 \
  --max-top-comments-per-video 20 \
  --max-replies-per-thread 5
```

Larger evaluation collection:

```bash
bash ./scripts/run/run_youtube_collection.sh \
  --run-label full_v2 \
  --max-keywords 20 \
  --max-videos-per-keyword 20 \
  --max-top-comments-per-video 100 \
  --max-replies-per-thread 30
```

After successful collection, check raw data:

```bash
find data/raw/youtube -type f | head
```

---

##  Run Bronze Layer Pipeline

Bronze layer stores raw data in a structured lakehouse format.

Run:

```bash
bash ./scripts/run/run_phase4_youtube_lakehouse.sh
```

Or, if your project has separate bronze script:

```bash
python -m pipelines.bronze.youtube_bronze_pipeline
```

Check output:

```bash
find data/bronze -maxdepth 3 -type d
```

---

## Run Silver Layer Pipeline

Silver layer cleans and standardizes YouTube comments.

Run:

```bash
python -m pipelines.silver.youtube_silver_pipeline
```

If you created a shell runner:

```bash
bash ./scripts/run/run_phase7_silver_pipeline.sh
```

Check output:

```bash
find data/silver -maxdepth 3 -type d
```

---

## Run Gold Analytics Mart Pipeline

Gold layer creates final marts used by the dashboard.

Run:

```bash
bash ./scripts/run/run_phase5_analytics_marts.sh
```

Or directly:

```bash
python -m pipelines.gold.youtube_gold_analytics_marts
```

Check output:

```bash
find data/gold -maxdepth 3 -type d
```

---

##  Run Descriptive Sentiment Marts

These marts are useful for dashboard KPIs and viva explanation.

Run:

```bash
python -m pipelines.gold.descriptive_sentiment_marts
```

Expected marts may include:

```text
topic_wise_sentiment_summary
daily_sentiment_trend_mart
hour_weekday_engagement_mart
keyword_frequency_mart
overview_kpi_mart
```

---

##  Run Diagnostic Insight Marts

Diagnostic marts explain why sentiment is positive or negative.

Run:

```bash
python -m pipelines.gold.diagnostic_sentiment_insights
```

Expected marts may include:

```text
negative_keyword_driver_mart
positive_keyword_driver_mart
sentiment_by_video_mart
engagement_vs_sentiment_mart
anomaly_spike_explanation_mart
```

---

##  Run Predictive Analytics Pipeline

Predictive analytics creates forecast and warning tables.

Run:

```bash
python -m pipelines.gold.predictive_analytics
```

Expected outputs may include:

```text
daily_comment_volume_forecast
sentiment_score_forecast
rising_topic_detection
negative_sentiment_warning_candidates
```

---

##  Run Prescriptive Analytics Pipeline

Prescriptive analytics creates recommended action tables.

Run:

```bash
python -m pipelines.gold.prescriptive_analytics
```

Expected outputs may include:

```text
recommended_actions_per_topic
alert_rules
monitoring_priority_score
business_recommendation_text
```

---

##  Run Full Pipeline in Correct Order

For final demo, run the pipeline in this order:

```bash
source .venv/bin/activate

bash ./scripts/run/run_youtube_collection.sh \
  --run-label demo_run \
  --max-keywords 3 \
  --max-videos-per-keyword 5 \
  --max-top-comments-per-video 20 \
  --max-replies-per-thread 5

bash ./scripts/run/run_phase4_youtube_lakehouse.sh
python -m pipelines.silver.youtube_silver_pipeline
bash ./scripts/run/run_phase5_analytics_marts.sh
python -m pipelines.gold.descriptive_sentiment_marts
python -m pipelines.gold.diagnostic_sentiment_insights
python -m pipelines.gold.predictive_analytics
python -m pipelines.gold.prescriptive_analytics
```

If some phase files have different names in your local repo, run the equivalent scripts from `scripts/run/`.

List available run scripts:

```bash
ls scripts/run
```

---

##  Check Dashboard Data Before Running UI

Run dashboard data check:

```bash
python -m scripts.dev.check_phase6_dashboard_data
```

If module import fails, run from project root and set `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD
python -m scripts.dev.check_phase6_dashboard_data
```

Expected result:

```text
Dashboard data loaded successfully
Gold marts available
```

---

##  Run Streamlit Dashboard

Run:

```bash
streamlit run dashboard/app.py
```

If your main dashboard file has a different name:

```bash
streamlit run dashboard/Home.py
```

Open the local URL shown in terminal, usually:

```text
http://localhost:8501
```

If running inside VM, use the VM IP address:

```text
http://<VM_IP_ADDRESS>:8501
```

Example:

```text
http://192.168.64.10:8501
```

---

##  Useful Git Commands

Check branch:

```bash
git branch
```

Check status:

```bash
git status
```

Create a new phase branch:

```bash
git switch -c feat/phase-xx-description
```

Add files:

```bash
git add .
```

Commit using project-style message:

```bash
git commit -m "feat(phase-xx): add description of completed work"
```

Push branch:

```bash
git push -u origin feat/phase-xx-description
```

Switch back to main:

```bash
git switch main
```

Merge feature branch into main:

```bash
git merge feat/phase-xx-description
```

Push main:

```bash
git push origin main
```

Important project rule:

```text
Do not delete phase branches unless the team explicitly decides to clean them later.
```

---

## Common Errors and Fixes

### Error: `ModuleNotFoundError: No module named 'common'`

Fix:

```bash
export PYTHONPATH=$PWD
```

Then rerun the command:

```bash
python -m scripts.dev.check_phase6_dashboard_data
```

---

### Error: `ModuleNotFoundError: No module named 'dashboard'`

Fix:

```bash
export PYTHONPATH=$PWD
streamlit run dashboard/app.py
```

---

### Error: `YOUTUBE_API_KEY not found`

Fix:

```bash
nano .env
```

Add:

```env
YOUTUBE_API_KEY=YOUR_REAL_API_KEY
```

Then rerun:

```bash
source .venv/bin/activate
```

---

### Error: Delta duplicate column found

Example:

```text
DELTA_DUPLICATE_COLUMNS_FOUND
```

Fix idea:

- Check the selected columns in the Gold pipeline.
- Make sure columns like `avg_comment_like_count` and `avg_comment_text_length` are not created twice.
- Drop duplicate columns before writing Delta output.

Useful debug command:

```bash
python -m pipelines.gold.youtube_gold_analytics_marts
```

---

### Spark warning: native Hadoop library

Warning:

```text
Unable to load native-hadoop library for your platform
```

Usually this is safe for local development. It does not mean the pipeline failed.

---

### Dashboard shows blank charts or NaN values

Run:

```bash
find data/gold -maxdepth 3 -type d
python -m scripts.dev.check_phase6_dashboard_data
```

Then rerun Gold pipelines:

```bash
bash ./scripts/run/run_phase5_analytics_marts.sh
python -m pipelines.gold.descriptive_sentiment_marts
python -m pipelines.gold.predictive_analytics
python -m pipelines.gold.prescriptive_analytics
```

---

##  Important Dashboard Metrics

The dashboard may show the following important metrics:

- Total comments
- Total videos analyzed
- Total topics / keywords
- Average sentiment score
- Positive comment percentage
- Negative comment percentage
- Neutral comment percentage
- Average comment likes
- Average comment text length
- Daily comment volume trend
- Daily sentiment trend
- Topic-wise sentiment summary
- Keyword frequency
- Engagement by hour and weekday
- Rising topics
- Negative sentiment warnings
- Recommended actions

---


##  One-Shot Demo Commands

Use this for quick final evaluation run:

```bash
cd ~/semester-2/AI528-BDA/term-project/social-pulse
source .venv/bin/activate
export PYTHONPATH=$PWD

python -m scripts.dev.check_spark_delta

bash ./scripts/run/run_youtube_collection.sh \
  --run-label demo_run \
  --max-keywords 3 \
  --max-videos-per-keyword 5 \
  --max-top-comments-per-video 20 \
  --max-replies-per-thread 5

bash ./scripts/run/run_phase4_youtube_lakehouse.sh
python -m pipelines.silver.youtube_silver_pipeline
bash ./scripts/run/run_phase5_analytics_marts.sh
python -m pipelines.gold.descriptive_sentiment_marts
python -m pipelines.gold.diagnostic_sentiment_insights
python -m pipelines.gold.predictive_analytics
python -m pipelines.gold.prescriptive_analytics
python -m scripts.dev.check_phase6_dashboard_data

streamlit run dashboard/app.py
```

---



### MongoDB Environment Variables

Add these values in `.env`:

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DATABASE=socialpulse
MONGO_YOUTUBE_COLLECTION=youtube_comments
MONGO_VIDEO_COLLECTION=youtube_videos
MONGO_RUN_COLLECTION=collection_runs
```

For MongoDB Atlas, replace `MONGO_URI` with your Atlas connection string. Do not commit the real URI to GitHub.

### Start MongoDB Locally with Docker

```bash
docker run -d \
  --name socialpulse-mongo \
  -p 27017:27017 \
  -v socialpulse_mongo_data:/data/db \
  mongo:7
```

If the container already exists:

```bash
docker start socialpulse-mongo
```

Check MongoDB is running:

```bash
docker ps | grep socialpulse-mongo
```

### Open MongoDB Shell

```bash
docker exec -it socialpulse-mongo mongosh
```

Inside MongoDB shell:

```javascript
show dbs
use socialpulse
show collections
```

### Verify MongoDB from Python

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD
pip install pymongo
```

```bash
python - <<'PYCODE'
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
db_name = os.getenv("MONGO_DATABASE", "socialpulse")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
client.admin.command("ping")
print("MongoDB connection OK")
print("Database:", db_name)
print("Collections:", client[db_name].list_collection_names())
client.close()
PYCODE
```

Expected output:

```text
MongoDB connection OK
Database: socialpulse
```

### MongoDB Collections Used

```text
socialpulse.youtube_comments  -> YouTube comments collected from API
socialpulse.youtube_videos    -> YouTube video metadata
socialpulse.collection_runs   -> collection run labels, timestamps, and status
socialpulse.pipeline_logs     -> optional pipeline execution logs
```

Useful verification commands:

```bash
docker exec -it socialpulse-mongo mongosh --eval '
use socialpulse;
print("comments = " + db.youtube_comments.countDocuments());
print("videos = " + db.youtube_videos.countDocuments());
print("runs = " + db.collection_runs.countDocuments());
printjson(db.youtube_comments.findOne());
'
```

### Create MongoDB Indexes

```bash
docker exec -it socialpulse-mongo mongosh --eval '
use socialpulse;
db.youtube_comments.createIndex({ video_id: 1 });
db.youtube_comments.createIndex({ keyword: 1 });
db.youtube_comments.createIndex({ published_at: 1 });
db.youtube_comments.createIndex({ sentiment_label: 1 });
db.youtube_videos.createIndex({ video_id: 1 }, { unique: true });
db.collection_runs.createIndex({ run_label: 1 });
'
```

### Run YouTube Collection with MongoDB Enabled

First make sure MongoDB is running and `.env` contains both YouTube and MongoDB values:

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD
docker start socialpulse-mongo
```

Then run collection:

```bash
bash ./scripts/run/run_youtube_collection.sh \
  --run-label demo_run \
  --max-keywords 3 \
  --max-videos-per-keyword 5 \
  --max-top-comments-per-video 20 \
  --max-replies-per-thread 5
```

Check inserted documents:

```bash
docker exec -it socialpulse-mongo mongosh --eval '
use socialpulse;
print("comments = " + db.youtube_comments.countDocuments());
print("videos = " + db.youtube_videos.countDocuments());
printjson(db.collection_runs.find().sort({created_at: -1}).limit(1).toArray());
'
```

### Export MongoDB Data for Lakehouse Raw Layer

If the Bronze pipeline reads raw JSON files from `data/raw/youtube`, export MongoDB collections like this:

```bash
mkdir -p data/raw/youtube/mongodb_export

docker exec socialpulse-mongo mongoexport \
  --db socialpulse \
  --collection youtube_comments \
  --out /tmp/youtube_comments.json \
  --jsonArray

docker cp socialpulse-mongo:/tmp/youtube_comments.json \
  data/raw/youtube/mongodb_export/youtube_comments.json
```

Export video metadata:

```bash
docker exec socialpulse-mongo mongoexport \
  --db socialpulse \
  --collection youtube_videos \
  --out /tmp/youtube_videos.json \
  --jsonArray

docker cp socialpulse-mongo:/tmp/youtube_videos.json \
  data/raw/youtube/mongodb_export/youtube_videos.json
```

Check files:

```bash
ls -lh data/raw/youtube/mongodb_export
head -n 5 data/raw/youtube/mongodb_export/youtube_comments.json
```

### Updated End-to-End Flow with MongoDB

```text
YouTube API
   ↓
MongoDB operational database
   ↓
Raw export / raw ingestion
   ↓
Bronze Delta layer
   ↓
Silver cleaned layer
   ↓
Gold analytics marts
   ↓
Streamlit dashboard
```
