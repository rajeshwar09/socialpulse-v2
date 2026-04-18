from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable

from socialpulse_v2.core.paths import LAKEHOUSE_ROOT


def _read_delta_table(path: Path) -> pd.DataFrame:
  if not path.exists():
    return pd.DataFrame()
  return DeltaTable(str(path)).to_pandas()


def _read_comment_level_data(gold_root: Path) -> pd.DataFrame:
  bronze_candidates = [
    LAKEHOUSE_ROOT / "bronze" / "youtube_comments_raw",
    LAKEHOUSE_ROOT / "bronze" / "youtube_comments_daily_raw",
  ]

  for candidate in bronze_candidates:
    if candidate.exists():
      try:
        comments_df = DeltaTable(str(candidate)).to_pandas()
        if not comments_df.empty:
          return comments_df
      except Exception:
        pass

  raw_daily_root = Path("data/raw/youtube/daily")
  if not raw_daily_root.exists():
    return pd.DataFrame()

  rows: list[dict] = []
  for file_path in sorted(raw_daily_root.glob("*/normalized_comments.json")):
    try:
      payload = json.loads(file_path.read_text(encoding="utf-8"))
      if isinstance(payload, list):
        rows.extend(payload)
    except Exception:
      continue

  if not rows:
    return pd.DataFrame()

  return pd.DataFrame(rows)


def _normalize_comment_columns(comments_df: pd.DataFrame) -> pd.DataFrame:
  if comments_df.empty:
    return comments_df

  df = comments_df.copy()

  rename_map = {
    "text": "comment_text",
    "published_at": "comment_published_at",
    "fetched_at": "collected_at",
    "source_run_id": "run_id",
  }
  for old_name, new_name in rename_map.items():
    if old_name in df.columns and new_name not in df.columns:
      df[new_name] = df[old_name]

  if "comment_published_at" in df.columns:
    df["comment_published_at"] = pd.to_datetime(df["comment_published_at"], errors="coerce", utc=True)
  else:
    df["comment_published_at"] = pd.NaT

  if "run_id" not in df.columns:
    df["run_id"] = pd.NA
  if "query_id" not in df.columns:
    df["query_id"] = pd.NA
  if "topic" not in df.columns:
    df["topic"] = pd.NA
  if "genre" not in df.columns:
    df["genre"] = pd.NA
  if "comment_text" not in df.columns:
    df["comment_text"] = pd.NA
  if "like_count" not in df.columns:
    df["like_count"] = 0

  df["like_count"] = pd.to_numeric(df["like_count"], errors="coerce").fillna(0)

  df["comment_date"] = df["comment_published_at"].dt.date
  df["comment_hour_24"] = df["comment_published_at"].dt.hour
  df["weekday_name"] = pd.Categorical(
    df["comment_published_at"].dt.day_name(),
    categories=[
      "Monday",
      "Tuesday",
      "Wednesday",
      "Thursday",
      "Friday",
      "Saturday",
      "Sunday",
    ],
    ordered=True,
  )

  return df


def load_dashboard_tables() -> dict[str, pd.DataFrame]:
  gold_root = LAKEHOUSE_ROOT / "gold"

  overview_df = _read_delta_table(gold_root / "dashboard_overview_daily")
  collection_df = _read_delta_table(gold_root / "collection_daily_summary")
  query_df = _read_delta_table(gold_root / "query_performance_summary")
  comments_df = _normalize_comment_columns(_read_comment_level_data(gold_root))

  sentiment_daily_summary_df = _read_delta_table(gold_root / "youtube_sentiment_daily_summary")
  sentiment_video_summary_df = _read_delta_table(gold_root / "youtube_sentiment_video_summary")
  sentiment_topic_summary_df = _read_delta_table(gold_root / "youtube_sentiment_topic_summary")
  sentiment_daily_trend_df = _read_delta_table(gold_root / "youtube_sentiment_daily_trend")
  sentiment_weekday_hour_df = _read_delta_table(gold_root / "youtube_sentiment_weekday_hour_engagement")
  sentiment_keyword_df = _read_delta_table(gold_root / "youtube_sentiment_keyword_frequency")
  sentiment_overview_kpis_df = _read_delta_table(gold_root / "youtube_sentiment_overview_kpis")

  sentiment_frames_with_date = [
    sentiment_daily_summary_df,
    sentiment_video_summary_df,
    sentiment_daily_trend_df,
    sentiment_weekday_hour_df,
    sentiment_keyword_df,
    sentiment_overview_kpis_df,
  ]

  for frame in sentiment_frames_with_date:
    if not frame.empty and "collection_date" in frame.columns:
      frame["collection_date"] = pd.to_datetime(frame["collection_date"], errors="coerce")
      frame["collection_date_label"] = frame["collection_date"].dt.strftime("%Y-%m-%d")

  standard_frames_with_run_date = [
    overview_df,
    collection_df,
    query_df,
  ]

  for frame in standard_frames_with_run_date:
    if not frame.empty and "run_date" in frame.columns:
      frame["run_date"] = pd.to_datetime(frame["run_date"], errors="coerce")
      frame["run_date_label"] = frame["run_date"].dt.strftime("%Y-%m-%d")

  numeric_columns = {
    "overview": [
      "topics_covered",
      "genres_covered",
      "queries_executed",
      "successful_queries",
      "partial_success_queries",
      "failed_queries",
      "no_data_queries",
      "unique_queries_seen",
      "total_videos_fetched",
      "total_records_fetched",
      "total_records_written",
      "total_error_count",
    ],
    "collection": [
      "queries_executed",
      "total_videos_fetched",
      "total_records_fetched",
      "total_records_written",
      "total_error_count",
    ],
    "query": [
      "expected_units",
      "videos_fetched",
      "records_fetched",
      "records_written",
      "error_count",
    ],
  }

  for column in numeric_columns["overview"]:
    if column in overview_df.columns:
      overview_df[column] = pd.to_numeric(overview_df[column], errors="coerce").fillna(0)

  for column in numeric_columns["collection"]:
    if column in collection_df.columns:
      collection_df[column] = pd.to_numeric(collection_df[column], errors="coerce").fillna(0)

  for column in numeric_columns["query"]:
    if column in query_df.columns:
      query_df[column] = pd.to_numeric(query_df[column], errors="coerce").fillna(0)

  return {
    "overview": overview_df,
    "collection": collection_df,
    "query": query_df,
    "comments": comments_df,
    "sentiment_daily_summary": sentiment_daily_summary_df,
    "sentiment_video_summary": sentiment_video_summary_df,
    "sentiment_topic_summary": sentiment_topic_summary_df,
    "sentiment_daily_trend": sentiment_daily_trend_df,
    "sentiment_weekday_hour_engagement": sentiment_weekday_hour_df,
    "sentiment_keyword_frequency": sentiment_keyword_df,
    "sentiment_overview_kpis": sentiment_overview_kpis_df,
  }


def apply_dashboard_filters(
  collection_df: pd.DataFrame,
  query_df: pd.DataFrame,
  selected_topics: list[str],
  selected_genres: list[str],
  selected_statuses: list[str],
  start_date,
  end_date,
  comments_df: pd.DataFrame | None = None,
):
  filtered_collection = collection_df.copy()
  filtered_query = query_df.copy()
  filtered_comments = comments_df.copy() if comments_df is not None else pd.DataFrame()

  if not filtered_collection.empty:
    if selected_topics:
      filtered_collection = filtered_collection[filtered_collection["topic"].isin(selected_topics)]
    if selected_genres:
      filtered_collection = filtered_collection[filtered_collection["genre"].isin(selected_genres)]
    if start_date is not None:
      filtered_collection = filtered_collection[filtered_collection["run_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
      filtered_collection = filtered_collection[filtered_collection["run_date"] <= pd.Timestamp(end_date)]

  if not filtered_query.empty:
    if selected_topics:
      filtered_query = filtered_query[filtered_query["topic"].isin(selected_topics)]
    if selected_genres:
      filtered_query = filtered_query[filtered_query["genre"].isin(selected_genres)]
    if selected_statuses:
      filtered_query = filtered_query[filtered_query["collection_status"].isin(selected_statuses)]
    if start_date is not None:
      filtered_query = filtered_query[filtered_query["run_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
      filtered_query = filtered_query[filtered_query["run_date"] <= pd.Timestamp(end_date)]

  if comments_df is not None and not filtered_comments.empty:
    if selected_topics and "topic" in filtered_comments.columns:
      filtered_comments = filtered_comments[filtered_comments["topic"].isin(selected_topics)]
    if selected_genres and "genre" in filtered_comments.columns:
      filtered_comments = filtered_comments[filtered_comments["genre"].isin(selected_genres)]
    if start_date is not None and "comment_published_at" in filtered_comments.columns:
      filtered_comments = filtered_comments[
        filtered_comments["comment_published_at"] >= pd.Timestamp(start_date).tz_localize("UTC")
      ]
    if end_date is not None and "comment_published_at" in filtered_comments.columns:
      filtered_comments = filtered_comments[
        filtered_comments["comment_published_at"] < (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize("UTC")
      ]

  if comments_df is None:
    return filtered_collection, filtered_query

  return filtered_collection, filtered_query, filtered_comments

def _filter_sentiment_frame(
  df: pd.DataFrame,
  selected_topics: list[str],
  selected_genres: list[str],
  start_date,
  end_date,
  date_column: str = "collection_date",
) -> pd.DataFrame:
  if df.empty:
    return df.copy()

  filtered = df.copy()

  if selected_topics and "topic" in filtered.columns:
    filtered = filtered[filtered["topic"].isin(selected_topics)]

  if selected_genres and "genre" in filtered.columns:
    filtered = filtered[filtered["genre"].isin(selected_genres)]

  if date_column in filtered.columns:
    filtered[date_column] = pd.to_datetime(filtered[date_column], errors="coerce")

    if start_date is not None:
      filtered = filtered[filtered[date_column] >= pd.Timestamp(start_date)]

    if end_date is not None:
      filtered = filtered[filtered[date_column] <= pd.Timestamp(end_date)]

  return filtered


def apply_sentiment_gold_filters(
  sentiment_daily_summary_df: pd.DataFrame,
  sentiment_video_summary_df: pd.DataFrame,
  sentiment_topic_summary_df: pd.DataFrame,
  sentiment_daily_trend_df: pd.DataFrame,
  sentiment_weekday_hour_df: pd.DataFrame,
  sentiment_keyword_df: pd.DataFrame,
  sentiment_overview_kpis_df: pd.DataFrame,
  selected_topics: list[str],
  selected_genres: list[str],
  start_date,
  end_date,
) -> dict[str, pd.DataFrame]:
  filtered_topic_summary = sentiment_topic_summary_df.copy()
  if not filtered_topic_summary.empty:
    if selected_topics and "topic" in filtered_topic_summary.columns:
      filtered_topic_summary = filtered_topic_summary[filtered_topic_summary["topic"].isin(selected_topics)]
    if selected_genres and "genre" in filtered_topic_summary.columns:
      filtered_topic_summary = filtered_topic_summary[filtered_topic_summary["genre"].isin(selected_genres)]

  return {
    "sentiment_daily_summary": _filter_sentiment_frame(
      sentiment_daily_summary_df, selected_topics, selected_genres, start_date, end_date
    ),
    "sentiment_video_summary": _filter_sentiment_frame(
      sentiment_video_summary_df, selected_topics, selected_genres, start_date, end_date
    ),
    "sentiment_topic_summary": filtered_topic_summary,
    "sentiment_daily_trend": _filter_sentiment_frame(
      sentiment_daily_trend_df, selected_topics, selected_genres, start_date, end_date
    ),
    "sentiment_weekday_hour_engagement": _filter_sentiment_frame(
      sentiment_weekday_hour_df, selected_topics, selected_genres, start_date, end_date
    ),
    "sentiment_keyword_frequency": _filter_sentiment_frame(
      sentiment_keyword_df, selected_topics, selected_genres, start_date, end_date
    ),
    "sentiment_overview_kpis": _filter_sentiment_frame(
      sentiment_overview_kpis_df, selected_topics, selected_genres, start_date, end_date
    ),
  }

def build_prescriptive_recommendations(query_df: pd.DataFrame) -> list[str]:
  recommendations: list[str] = []

  if query_df.empty:
    return ["No query performance data is available yet, so no recommendations can be generated."]

  working_df = query_df.copy()

  if "expected_units" not in working_df.columns:
    working_df["expected_units"] = 0
  if "records_written" not in working_df.columns:
    working_df["records_written"] = 0
  if "collection_status" not in working_df.columns:
    working_df["collection_status"] = "unknown"
  if "topic" not in working_df.columns:
    working_df["topic"] = "unknown"
  if "query_id" not in working_df.columns:
    working_df["query_id"] = "unknown"

  working_df["expected_units"] = pd.to_numeric(working_df["expected_units"], errors="coerce").fillna(0)
  working_df["records_written"] = pd.to_numeric(working_df["records_written"], errors="coerce").fillna(0)

  working_df["efficiency_ratio"] = working_df.apply(
    lambda row: (row["records_written"] / row["expected_units"]) if row["expected_units"] > 0 else 0,
    axis=1,
  )

  no_data_df = working_df[working_df["collection_status"] == "no_data"]
  failed_df = working_df[working_df["collection_status"] == "failed"]
  low_efficiency_df = working_df[working_df["efficiency_ratio"] < 0.25]

  if not no_data_df.empty:
    top_no_data = (
      no_data_df.groupby("topic", as_index=False)["query_id"]
      .count()
      .rename(columns={"query_id": "count"}) # type: ignore
      .sort_values("count", ascending=False)
      .head(3)
    )
    for _, row in top_no_data.iterrows():
      recommendations.append(
        f"Topic '{row['topic']}' produced no data in {int(row['count'])} query runs. Consider refining the query wording, broadening the search intent, or increasing the lookback window."
      )

  if not failed_df.empty:
    top_failed = (
      failed_df.groupby("topic", as_index=False)["query_id"]
      .count()
      .rename(columns={"query_id": "count"}) # type: ignore
      .sort_values("count", ascending=False)
      .head(3)
    )
    for _, row in top_failed.iterrows():
      recommendations.append(
        f"Topic '{row['topic']}' had {int(row['count'])} failed query runs. Review API quota, request retry handling, and platform-side failures for this topic."
      )

  if "expected_units" in query_df.columns and not low_efficiency_df.empty:
    weak_queries = (
      low_efficiency_df.sort_values(["efficiency_ratio", "records_written"], ascending=[True, True])
      [["query_id", "topic", "efficiency_ratio"]]
      .drop_duplicates()
      .head(3)
    )
    for _, row in weak_queries.iterrows():
      recommendations.append(
        f"Query '{row['query_id']}' underperformed for topic '{row['topic']}' with a low fulfillment ratio of {row['efficiency_ratio']:.2f}. This query should be rewritten or deprioritized."
      )

  strong_topics = (
    working_df.groupby("topic", as_index=False)["records_written"]
    .sum()
    .sort_values("records_written", ascending=False) # type: ignore
    .head(3)
  )

  for _, row in strong_topics.iterrows():
    recommendations.append(
      f"Topic '{row['topic']}' is currently one of the strongest contributors with {int(row['records_written'])} written records. Keep it in the stable daily query portfolio."
    )

  unique_recommendations = []
  seen = set()

  for item in recommendations:
    if item not in seen:
      seen.add(item)
      unique_recommendations.append(item)

  if not unique_recommendations:
    unique_recommendations.append(
      "The current collection looks stable. The next step is to increase query diversity and historical depth before training predictive or prescriptive models."
    )

  return unique_recommendations[:6]