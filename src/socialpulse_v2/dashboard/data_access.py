from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from deltalake import DeltaTable

from socialpulse_v2.core.paths import LAKEHOUSE_ROOT


ALIAS_PATH = Path("configs/topic_aliases.json")

WEEKDAY_ORDER = [
  "Monday",
  "Tuesday",
  "Wednesday",
  "Thursday",
  "Friday",
  "Saturday",
  "Sunday",
]


def _read_delta_table(path: Path) -> pd.DataFrame:
  if not path.exists():
    return pd.DataFrame()
  return DeltaTable(str(path)).to_pandas()


def load_topic_aliases() -> dict[str, list[str]]:
  if not ALIAS_PATH.exists():
    return {}
  return json.loads(ALIAS_PATH.read_text(encoding="utf-8"))


def resolve_analysis_query(query: str) -> dict[str, str | list[str] | None]:
  cleaned = query.strip().lower()
  if not cleaned:
    return {
      "raw_query": "",
      "matched_topic": None,
      "matched_genre": None,
      "matched_aliases": [],
    }

  alias_map = load_topic_aliases()
  matched_topic = None
  matched_genre = None
  matched_aliases: list[str] = []

  for topic, aliases in alias_map.items():
    alias_list = aliases if isinstance(aliases, list) else []
    for alias in alias_list:
      alias_clean = str(alias).strip().lower()
      if alias_clean and (
        cleaned == alias_clean
        or cleaned in alias_clean
        or alias_clean in cleaned
      ):
        matched_topic = str(topic).strip().lower()
        matched_aliases.append(str(alias))
        break
    if matched_topic:
      break

  if matched_topic is not None:
    query_registry_path = Path("configs/query_registry.json")
    if query_registry_path.exists():
      try:
        payload = json.loads(query_registry_path.read_text(encoding="utf-8"))
        for row in payload:
          if str(row.get("topic", "")).strip().lower() == matched_topic:
            matched_genre = str(row.get("genre", "")).strip().lower() or None
            break
      except Exception:
        matched_genre = None

  return {
    "raw_query": cleaned,
    "matched_topic": matched_topic,
    "matched_genre": matched_genre,
    "matched_aliases": matched_aliases,
  }


def _normalize_text_series(df: pd.DataFrame, column: str) -> pd.Series:
  if column not in df.columns:
    return pd.Series([""] * len(df), index=df.index)
  return df[column].fillna("").astype(str).str.strip().str.lower()


def _exact_topic_genre_mask(df: pd.DataFrame, cleaned_query: str) -> pd.Series:
  if df.empty:
    return pd.Series(dtype="bool")

  mask = pd.Series([False] * len(df), index=df.index)

  if "topic" in df.columns:
    mask = mask | _normalize_text_series(df, "topic").eq(cleaned_query)

  if "genre" in df.columns:
    mask = mask | _normalize_text_series(df, "genre").eq(cleaned_query)

  return mask


def _query_text_mask(df: pd.DataFrame, cleaned_query: str) -> pd.Series:
  if df.empty:
    return pd.Series(dtype="bool")

  mask = pd.Series([False] * len(df), index=df.index)

  if "query_text" in df.columns:
    mask = mask | _normalize_text_series(df, "query_text").str.contains(
      cleaned_query,
      regex=False,
    )

  return mask


def _alias_context_mask(
  df: pd.DataFrame,
  matched_topic: str | None,
  matched_genre: str | None,
) -> pd.Series:
  if df.empty:
    return pd.Series(dtype="bool")

  mask = pd.Series([False] * len(df), index=df.index)

  if matched_topic and "topic" in df.columns:
    mask = mask | _normalize_text_series(df, "topic").eq(str(matched_topic).lower())

  if matched_genre and "genre" in df.columns:
    mask = mask | _normalize_text_series(df, "genre").eq(str(matched_genre).lower())

  return mask


def _determine_query_scope_mode(
  cleaned_query: str,
  matched_topic: str | None,
  matched_genre: str | None,
  filtered_comments: pd.DataFrame,
  filtered_collection: pd.DataFrame,
  filtered_query: pd.DataFrame,
) -> str | None:
  candidates = [
    filtered_comments,
    filtered_collection,
    filtered_query,
  ]

  for candidate in candidates:
    if candidate.empty:
      continue
    exact_mask = _exact_topic_genre_mask(candidate, cleaned_query)
    if not exact_mask.empty and exact_mask.any():
      return "exact_topic_or_genre"

  for candidate in candidates:
    if candidate.empty:
      continue
    query_text_mask = _query_text_mask(candidate, cleaned_query)
    if not query_text_mask.empty and query_text_mask.any():
      return "query_text"

  if matched_topic or matched_genre:
    return "alias_context"

  return None


def _apply_scope_to_frame(
  df: pd.DataFrame,
  scope_mode: str | None,
  cleaned_query: str,
  matched_topic: str | None,
  matched_genre: str | None,
) -> pd.DataFrame:
  if df.empty:
    return df.copy()

  if scope_mode == "exact_topic_or_genre":
    mask = _exact_topic_genre_mask(df, cleaned_query)
  elif scope_mode == "query_text":
    mask = _query_text_mask(df, cleaned_query)
  elif scope_mode == "alias_context":
    mask = _alias_context_mask(df, matched_topic, matched_genre)
  else:
    mask = pd.Series([True] * len(df), index=df.index)

  if mask.empty or not mask.any():
    return df.iloc[0:0].copy()

  return df[mask].copy()


def _read_comment_level_data() -> pd.DataFrame:
  preferred_candidates = [
    LAKEHOUSE_ROOT / "silver" / "youtube_comments_sentiment",
    LAKEHOUSE_ROOT / "bronze" / "youtube_comments_daily_raw",
    LAKEHOUSE_ROOT / "bronze" / "youtube_comments_raw",
  ]

  for candidate in preferred_candidates:
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
    "like_count": "comment_like_count",
    "author_name": "author_display_name",
  }

  for old_name, new_name in rename_map.items():
    if old_name in df.columns and new_name not in df.columns:
      df[new_name] = df[old_name]

  defaults = {
    "run_id": pd.NA,
    "query_id": pd.NA,
    "query_text": pd.NA,
    "topic": pd.NA,
    "genre": pd.NA,
    "video_id": pd.NA,
    "video_title": pd.NA,
    "channel_title": pd.NA,
    "comment_id": pd.NA,
    "comment_text": pd.NA,
    "comment_like_count": 0,
    "reply_count": 0,
    "sentiment_label": pd.NA,
    "sentiment_score": 0.0,
    "platform": "youtube",
  }

  for column, default_value in defaults.items():
    if column not in df.columns:
      df[column] = default_value

  if "comment_published_at" in df.columns:
    df["comment_published_at"] = pd.to_datetime(
      df["comment_published_at"],
      errors="coerce",
      utc=True,
    )
  else:
    df["comment_published_at"] = pd.NaT

  if "collection_date" not in df.columns:
    if "comment_published_at" in df.columns:
      df["collection_date"] = (
        pd.to_datetime(df["comment_published_at"], errors="coerce")
        .dt.date
        .astype("string")
      )
    else:
      df["collection_date"] = pd.NA

  df["comment_like_count"] = pd.to_numeric(
    df["comment_like_count"],
    errors="coerce",
  ).fillna(0).astype("int64")

  df["reply_count"] = pd.to_numeric(
    df["reply_count"],
    errors="coerce",
  ).fillna(0).astype("int64")

  df["sentiment_score"] = pd.to_numeric(
    df["sentiment_score"],
    errors="coerce",
  ).fillna(0.0).astype("float64")

  df["comment_hour_24"] = pd.to_numeric(
    df["comment_published_at"].dt.hour,
    errors="coerce",
  ).astype("Int64")

  df["weekday_name"] = pd.Categorical(
    df["comment_published_at"].dt.day_name(),
    categories=WEEKDAY_ORDER,
    ordered=True,
  )

  return df


def _align_timestamp_to_series_tz(series: pd.Series, value) -> pd.Timestamp:
  ts = pd.Timestamp(value)

  tz = None
  try:
    tz = series.dt.tz
  except Exception:
    tz = None

  if tz is not None:
    if ts.tzinfo is None:
      ts = ts.tz_localize(tz)
    else:
      ts = ts.tz_convert(tz)
  else:
    if ts.tzinfo is not None:
      ts = ts.tz_localize(None)

  return ts


def load_dashboard_tables() -> dict[str, pd.DataFrame]:
  gold_root = LAKEHOUSE_ROOT / "gold"

  overview_df = _read_delta_table(gold_root / "dashboard_overview_daily")
  collection_df = _read_delta_table(gold_root / "collection_daily_summary")
  query_df = _read_delta_table(gold_root / "query_performance_summary")
  sentiment_comments_df = _normalize_comment_columns(_read_comment_level_data())

  sentiment_daily_summary_df = _read_delta_table(gold_root / "youtube_sentiment_daily_summary")
  sentiment_video_summary_df = _read_delta_table(gold_root / "youtube_sentiment_video_summary")
  sentiment_topic_summary_df = _read_delta_table(gold_root / "youtube_sentiment_topic_summary")
  sentiment_daily_trend_df = _read_delta_table(gold_root / "youtube_sentiment_daily_trend")
  sentiment_weekday_hour_df = _read_delta_table(gold_root / "youtube_sentiment_weekday_hour_engagement")
  sentiment_keyword_df = _read_delta_table(gold_root / "youtube_sentiment_keyword_frequency")
  sentiment_overview_kpis_df = _read_delta_table(gold_root / "youtube_sentiment_overview_kpis")

  predictive_forecast_summary_df = _read_delta_table(gold_root / "youtube_comments_forecast_summary")
  predictive_forecast_7d_df = _read_delta_table(gold_root / "youtube_comments_forecast_7d")

  for frame in [overview_df, collection_df, query_df]:
    if not frame.empty and "run_date" in frame.columns:
      frame["run_date"] = pd.to_datetime(frame["run_date"], errors="coerce")
      frame["run_date_label"] = frame["run_date"].dt.strftime("%Y-%m-%d")

  for frame in [
    sentiment_daily_summary_df,
    sentiment_video_summary_df,
    sentiment_daily_trend_df,
    sentiment_weekday_hour_df,
    sentiment_keyword_df,
    sentiment_overview_kpis_df,
  ]:
    if not frame.empty and "collection_date" in frame.columns:
      frame["collection_date"] = pd.to_datetime(frame["collection_date"], errors="coerce")
      frame["collection_date_label"] = frame["collection_date"].dt.strftime("%Y-%m-%d")

  for frame in [sentiment_topic_summary_df]:
    if not frame.empty and "built_at" in frame.columns:
      frame["built_at"] = pd.to_datetime(frame["built_at"], errors="coerce")

  for frame in [predictive_forecast_summary_df]:
    if not frame.empty:
      for column in ["history_start_date", "history_end_date", "built_at"]:
        if column in frame.columns:
          frame[column] = pd.to_datetime(frame[column], errors="coerce")

  for frame in [predictive_forecast_7d_df]:
    if not frame.empty:
      for column in ["forecast_date", "built_at"]:
        if column in frame.columns:
          frame[column] = pd.to_datetime(frame[column], errors="coerce")

  return {
    "overview": overview_df,
    "collection": collection_df,
    "query": query_df,
    "comments": sentiment_comments_df,
    "sentiment_comments": sentiment_comments_df,
    "sentiment_daily_summary": sentiment_daily_summary_df,
    "sentiment_video_summary": sentiment_video_summary_df,
    "sentiment_topic_summary": sentiment_topic_summary_df,
    "sentiment_daily_trend": sentiment_daily_trend_df,
    "sentiment_weekday_hour_engagement": sentiment_weekday_hour_df,
    "sentiment_keyword_frequency": sentiment_keyword_df,
    "sentiment_overview_kpis": sentiment_overview_kpis_df,
    "predictive_forecast_summary": predictive_forecast_summary_df,
    "predictive_forecast_7d": predictive_forecast_7d_df,
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
  analysis_query: str | None = None,
):
  filtered_collection = collection_df.copy()
  filtered_query = query_df.copy()
  filtered_comments = comments_df.copy() if comments_df is not None else pd.DataFrame()

  if not filtered_collection.empty:
    if selected_topics and "topic" in filtered_collection.columns:
      filtered_collection = filtered_collection[filtered_collection["topic"].isin(selected_topics)]
    if selected_genres and "genre" in filtered_collection.columns:
      filtered_collection = filtered_collection[filtered_collection["genre"].isin(selected_genres)]
    if start_date is not None and "run_date" in filtered_collection.columns:
      filtered_collection = filtered_collection[
        filtered_collection["run_date"] >= pd.Timestamp(start_date)
      ]
    if end_date is not None and "run_date" in filtered_collection.columns:
      filtered_collection = filtered_collection[
        filtered_collection["run_date"] <= pd.Timestamp(end_date)
      ]

  if not filtered_query.empty:
    if selected_topics and "topic" in filtered_query.columns:
      filtered_query = filtered_query[filtered_query["topic"].isin(selected_topics)]
    if selected_genres and "genre" in filtered_query.columns:
      filtered_query = filtered_query[filtered_query["genre"].isin(selected_genres)]
    if selected_statuses and "collection_status" in filtered_query.columns:
      filtered_query = filtered_query[filtered_query["collection_status"].isin(selected_statuses)]
    if start_date is not None and "run_date" in filtered_query.columns:
      filtered_query = filtered_query[
        filtered_query["run_date"] >= pd.Timestamp(start_date)
      ]
    if end_date is not None and "run_date" in filtered_query.columns:
      filtered_query = filtered_query[
        filtered_query["run_date"] <= pd.Timestamp(end_date)
      ]

  analysis_context = resolve_analysis_query(analysis_query or "")
  cleaned_query = str(analysis_context["raw_query"] or "").strip().lower()
  matched_topic = analysis_context["matched_topic"]
  matched_genre = analysis_context["matched_genre"]

  if comments_df is not None and not filtered_comments.empty:
    if selected_topics and "topic" in filtered_comments.columns:
      filtered_comments = filtered_comments[filtered_comments["topic"].isin(selected_topics)]
    if selected_genres and "genre" in filtered_comments.columns:
      filtered_comments = filtered_comments[filtered_comments["genre"].isin(selected_genres)]

    if start_date is not None and "comment_published_at" in filtered_comments.columns:
      start_ts = pd.Timestamp(start_date).tz_localize("UTC")
      filtered_comments = filtered_comments[
        filtered_comments["comment_published_at"] >= start_ts
      ]

    if end_date is not None and "comment_published_at" in filtered_comments.columns:
      end_ts = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize("UTC")
      filtered_comments = filtered_comments[
        filtered_comments["comment_published_at"] < end_ts
      ]

  if cleaned_query:
    scope_mode = _determine_query_scope_mode(
      cleaned_query=cleaned_query,
      matched_topic=matched_topic,
      matched_genre=matched_genre,
      filtered_comments=filtered_comments,
      filtered_collection=filtered_collection,
      filtered_query=filtered_query,
    )

    filtered_collection = _apply_scope_to_frame(
      filtered_collection,
      scope_mode,
      cleaned_query,
      matched_topic,
      matched_genre,
    )
    filtered_query = _apply_scope_to_frame(
      filtered_query,
      scope_mode,
      cleaned_query,
      matched_topic,
      matched_genre,
    )

    if comments_df is not None:
      filtered_comments = _apply_scope_to_frame(
        filtered_comments,
        scope_mode,
        cleaned_query,
        matched_topic,
        matched_genre,
      )

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
  analysis_query: str | None = None,
  filtered_sentiment_comments: pd.DataFrame | None = None,
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
      start_ts = _align_timestamp_to_series_tz(filtered[date_column], start_date)
      filtered = filtered[filtered[date_column] >= start_ts]

    if end_date is not None:
      end_ts = _align_timestamp_to_series_tz(
        filtered[date_column],
        pd.Timestamp(end_date) + pd.Timedelta(days=1),
      )
      filtered = filtered[filtered[date_column] < end_ts]

  if filtered_sentiment_comments is not None and not filtered_sentiment_comments.empty:
    if "topic" in filtered.columns and "topic" in filtered_sentiment_comments.columns:
      allowed_topics = (
        filtered_sentiment_comments["topic"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
      )
      if allowed_topics:
        filtered = filtered[filtered["topic"].astype(str).isin(allowed_topics)]

    if "genre" in filtered.columns and "genre" in filtered_sentiment_comments.columns:
      allowed_genres = (
        filtered_sentiment_comments["genre"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
      )
      if allowed_genres:
        filtered = filtered[filtered["genre"].astype(str).isin(allowed_genres)]

    if "video_id" in filtered.columns and "video_id" in filtered_sentiment_comments.columns:
      allowed_video_ids = (
        filtered_sentiment_comments["video_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
      )
      if allowed_video_ids:
        filtered = filtered[filtered["video_id"].astype(str).isin(allowed_video_ids)]

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
  analysis_query: str | None = None,
  filtered_sentiment_comments: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
  return {
    "sentiment_daily_summary": _filter_sentiment_frame(
      sentiment_daily_summary_df,
      selected_topics,
      selected_genres,
      start_date,
      end_date,
      analysis_query=analysis_query,
      filtered_sentiment_comments=filtered_sentiment_comments,
    ),
    "sentiment_video_summary": _filter_sentiment_frame(
      sentiment_video_summary_df,
      selected_topics,
      selected_genres,
      start_date,
      end_date,
      analysis_query=analysis_query,
      filtered_sentiment_comments=filtered_sentiment_comments,
    ),
    "sentiment_topic_summary": _filter_sentiment_frame(
      sentiment_topic_summary_df,
      selected_topics,
      selected_genres,
      start_date,
      end_date,
      date_column="built_at",
      analysis_query=analysis_query,
      filtered_sentiment_comments=filtered_sentiment_comments,
    ),
    "sentiment_daily_trend": _filter_sentiment_frame(
      sentiment_daily_trend_df,
      selected_topics,
      selected_genres,
      start_date,
      end_date,
      analysis_query=analysis_query,
      filtered_sentiment_comments=filtered_sentiment_comments,
    ),
    "sentiment_weekday_hour_engagement": _filter_sentiment_frame(
      sentiment_weekday_hour_df,
      selected_topics,
      selected_genres,
      start_date,
      end_date,
      analysis_query=analysis_query,
      filtered_sentiment_comments=filtered_sentiment_comments,
    ),
    "sentiment_keyword_frequency": _filter_sentiment_frame(
      sentiment_keyword_df,
      selected_topics,
      selected_genres,
      start_date,
      end_date,
      analysis_query=analysis_query,
      filtered_sentiment_comments=filtered_sentiment_comments,
    ),
    "sentiment_overview_kpis": _filter_sentiment_frame(
      sentiment_overview_kpis_df,
      selected_topics,
      selected_genres,
      start_date,
      end_date,
      analysis_query=analysis_query,
      filtered_sentiment_comments=filtered_sentiment_comments,
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

  working_df["expected_units"] = pd.to_numeric(
    working_df["expected_units"],
    errors="coerce",
  ).fillna(0)

  working_df["records_written"] = pd.to_numeric(
    working_df["records_written"],
    errors="coerce",
  ).fillna(0)

  working_df["efficiency_ratio"] = working_df.apply(
    lambda row: (row["records_written"] / row["expected_units"])
    if row["expected_units"] > 0
    else 0,
    axis=1,
  )

  weak_queries = working_df.sort_values(
    ["efficiency_ratio", "records_written"],
    ascending=[True, True],
  ).head(3)

  for _, row in weak_queries.iterrows():
    recommendations.append(
      f"Query '{row['query_id']}' is underperforming for topic '{row['topic']}'. Rewrite or deprioritize it."
    )

  if not recommendations:
    recommendations.append(
      "The current collection looks stable. Increase history depth before retraining forecasts."
    )

  return recommendations[:6]