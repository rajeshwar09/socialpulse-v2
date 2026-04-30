from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


STOPWORDS = {
  "the", "and", "for", "that", "this", "with", "you", "your", "are", "was", "were",
  "have", "has", "had", "from", "they", "their", "but", "not", "too", "very", "all",
  "can", "just", "out", "about", "into", "what", "when", "where", "how", "why",
  "will", "would", "could", "should", "there", "here", "then", "than", "them",
  "his", "her", "she", "him", "our", "its", "it's", "dont", "didnt", "doesnt",
  "im", "ive", "ill", "one", "two", "get", "got", "really", "also", "because",
  "after", "before", "over", "under", "more", "most", "much", "many", "some",
  "movie", "movies", "video", "review", "shorts", "youtube", "watch", "watching",
}


def _pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


def _shorten(text: object, max_len: int = 58) -> str:
  value = str(text)
  return value if len(value) <= max_len else value[: max_len - 3] + "..."


def _safe_divide(numerator: float, denominator: float) -> float:
  if denominator == 0:
    return 0.0
  return numerator / denominator


def _weighted_average(df: pd.DataFrame, value_col: str, weight_col: str) -> float:
  if df.empty or value_col not in df.columns or weight_col not in df.columns:
    return 0.0

  working = df[[value_col, weight_col]].copy()
  working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
  working[weight_col] = pd.to_numeric(working[weight_col], errors="coerce")
  working = working.dropna()

  if working.empty:
    return 0.0

  total_weight = float(working[weight_col].sum())
  if total_weight <= 0:
    return float(working[value_col].mean())

  return float((working[value_col] * working[weight_col]).sum() / total_weight)


def _resolve_like_count_column(df: pd.DataFrame) -> str | None:
  candidates = [
    "comment_like_count",
    "like_count",
  ]
  for col in candidates:
    if col in df.columns:
      return col
  return None


def _tokenize(text: object) -> list[str]:
  if text is None or pd.isna(text):
    return []

  tokens = re.findall(r"[a-z0-9]+", str(text).lower())
  return [
    token for token in tokens
    if len(token) >= 3 and token not in STOPWORDS and not token.isdigit()
  ]


def _build_topic_summary_from_comments(filtered_sentiment_comments: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_comments.empty:
    return pd.DataFrame()

  working = filtered_sentiment_comments.copy()

  if "topic" not in working.columns:
    working["topic"] = "unknown"
  if "genre" not in working.columns:
    working["genre"] = "unknown"
  if "comment_id" not in working.columns:
    working["comment_id"] = range(len(working))
  if "video_id" not in working.columns:
    working["video_id"] = ""
  if "sentiment_score" not in working.columns:
    working["sentiment_score"] = 0.0
  if "sentiment_label" not in working.columns:
    working["sentiment_label"] = "neutral"

  like_col = _resolve_like_count_column(working)
  if like_col is None:
    working["__resolved_like_count"] = 0.0
  else:
    working["__resolved_like_count"] = pd.to_numeric(
      working[like_col],
      errors="coerce",
    ).fillna(0.0)

  grouped = (
    working.groupby(["topic", "genre"], as_index=False)
    .agg(
      comments_count=("comment_id", "nunique"),
      videos_covered=("video_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
      negative_comments=("sentiment_label", lambda s: int((s.astype(str) == "negative").sum())),
      positive_comments=("sentiment_label", lambda s: int((s.astype(str) == "positive").sum())),
      total_comment_likes=("__resolved_like_count", "sum"),
      avg_comment_likes=("__resolved_like_count", "mean"),
    )
  )

  denominator = grouped["comments_count"].replace(0, 1)
  grouped["negative_ratio"] = grouped["negative_comments"] / denominator
  grouped["positive_ratio"] = grouped["positive_comments"] / denominator
  grouped["risk_load"] = grouped["comments_count"] * grouped["negative_ratio"]
  grouped["topic_display"] = grouped["topic"].map(_pretty_text)
  grouped["genre_display"] = grouped["genre"].map(_pretty_text)

  return grouped.sort_values(["risk_load", "comments_count"], ascending=[False, False])


def _build_video_summary_from_comments(filtered_sentiment_comments: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_comments.empty:
    return pd.DataFrame()

  working = filtered_sentiment_comments.copy()

  if "video_id" not in working.columns:
    working["video_id"] = ""
  if "video_title" not in working.columns:
    working["video_title"] = ""
  if "channel_title" not in working.columns:
    working["channel_title"] = ""
  if "topic" not in working.columns:
    working["topic"] = "unknown"
  if "genre" not in working.columns:
    working["genre"] = "unknown"
  if "comment_id" not in working.columns:
    working["comment_id"] = range(len(working))
  if "sentiment_score" not in working.columns:
    working["sentiment_score"] = 0.0
  if "sentiment_label" not in working.columns:
    working["sentiment_label"] = "neutral"

  like_col = _resolve_like_count_column(working)
  if like_col is None:
    working["__resolved_like_count"] = 0.0
  else:
    working["__resolved_like_count"] = pd.to_numeric(
      working[like_col],
      errors="coerce",
    ).fillna(0.0)

  grouped = (
    working.groupby(
      ["video_id", "video_title", "channel_title", "topic", "genre"],
      as_index=False,
    )
    .agg(
      comments_count=("comment_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
      negative_comments=("sentiment_label", lambda s: int((s.astype(str) == "negative").sum())),
      positive_comments=("sentiment_label", lambda s: int((s.astype(str) == "positive").sum())),
      total_comment_likes=("__resolved_like_count", "sum"),
      avg_comment_likes=("__resolved_like_count", "mean"),
    )
  )

  denominator = grouped["comments_count"].replace(0, 1)
  grouped["negative_ratio"] = grouped["negative_comments"] / denominator
  grouped["positive_ratio"] = grouped["positive_comments"] / denominator
  grouped["video_display"] = grouped["video_title"].map(_shorten)
  grouped["topic_display"] = grouped["topic"].map(_pretty_text)
  grouped["genre_display"] = grouped["genre"].map(_pretty_text)

  return grouped


def _build_keyword_driver_df_from_gold(
  filtered_sentiment_keyword: pd.DataFrame,
  positive: bool,
  top_n: int = 15,
) -> pd.DataFrame:
  columns = [
    "topic",
    "genre",
    "keyword",
    "keyword_count",
    "distinct_comments",
    "avg_sentiment_score",
    "topic_display",
    "genre_display",
  ]

  if filtered_sentiment_keyword.empty:
    return pd.DataFrame(columns=columns)

  working = filtered_sentiment_keyword.copy()

  required = ["topic", "genre", "keyword", "keyword_count", "distinct_comments", "avg_sentiment_score"]
  for col in required:
    if col not in working.columns:
      if col in {"keyword_count", "distinct_comments", "avg_sentiment_score"}:
        working[col] = 0
      else:
        working[col] = ""

  working["keyword_count"] = pd.to_numeric(working["keyword_count"], errors="coerce").fillna(0)
  working["distinct_comments"] = pd.to_numeric(working["distinct_comments"], errors="coerce").fillna(0)
  working["avg_sentiment_score"] = pd.to_numeric(working["avg_sentiment_score"], errors="coerce").fillna(0)

  if positive:
    working = working[working["avg_sentiment_score"] > 0]
  else:
    working = working[working["avg_sentiment_score"] < 0]

  if working.empty:
    return pd.DataFrame(columns=columns)

  def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    return pd.Series(
      {
        "keyword_count": float(group["keyword_count"].sum()),
        "distinct_comments": float(group["distinct_comments"].sum()),
        "avg_sentiment_score": _weighted_average(
          group,
          "avg_sentiment_score",
          "distinct_comments" if float(group["distinct_comments"].sum()) > 0 else "keyword_count",
        ),
      }
    )

  grouped = (
    working.groupby(["topic", "genre", "keyword"])
    .apply(_aggregate_group)
    .reset_index()
    .sort_values(["keyword_count", "distinct_comments"], ascending=[False, False])
    .head(top_n)
  )

  grouped["topic_display"] = grouped["topic"].map(_pretty_text)
  grouped["genre_display"] = grouped["genre"].map(_pretty_text)

  return grouped[columns]


def _build_keyword_driver_df_from_comments(
  filtered_sentiment_comments: pd.DataFrame,
  positive: bool,
  top_n: int = 15,
) -> pd.DataFrame:
  columns = [
    "topic",
    "genre",
    "keyword",
    "keyword_count",
    "distinct_comments",
    "avg_sentiment_score",
    "topic_display",
    "genre_display",
  ]

  if filtered_sentiment_comments.empty:
    return pd.DataFrame(columns=columns)

  working = filtered_sentiment_comments.copy()

  if "comment_id" not in working.columns:
    working["comment_id"] = range(len(working))
  if "comment_text" not in working.columns:
    working["comment_text"] = ""
  if "topic" not in working.columns:
    working["topic"] = "unknown"
  if "genre" not in working.columns:
    working["genre"] = "unknown"
  if "sentiment_score" not in working.columns:
    working["sentiment_score"] = 0.0

  if positive:
    working = working[working["sentiment_score"] > 0]
  else:
    working = working[working["sentiment_score"] < 0]

  if working.empty:
    return pd.DataFrame(columns=columns)

  keyword_counter: Counter[str] = Counter()
  keyword_comment_ids: dict[str, set[str]] = {}
  keyword_scores: dict[str, list[float]] = {}
  keyword_topic: dict[str, str] = {}
  keyword_genre: dict[str, str] = {}

  for _, row in working.iterrows():
    comment_id = str(row.get("comment_id", ""))
    score = float(row.get("sentiment_score", 0.0))
    topic = str(row.get("topic", "unknown"))
    genre = str(row.get("genre", "unknown"))

    unique_keywords = set(_tokenize(row.get("comment_text", "")))
    for keyword in unique_keywords:
      keyword_counter[keyword] += 1
      keyword_comment_ids.setdefault(keyword, set()).add(comment_id)
      keyword_scores.setdefault(keyword, []).append(score)
      keyword_topic.setdefault(keyword, topic)
      keyword_genre.setdefault(keyword, genre)

  rows: list[dict] = []
  for keyword, keyword_count in keyword_counter.most_common(top_n):
    rows.append(
      {
        "topic": keyword_topic.get(keyword, "unknown"),
        "genre": keyword_genre.get(keyword, "unknown"),
        "keyword": keyword,
        "keyword_count": int(keyword_count),
        "distinct_comments": int(len(keyword_comment_ids.get(keyword, set()))),
        "avg_sentiment_score": round(float(pd.Series(keyword_scores.get(keyword, [0.0])).mean()), 4),
      }
    )

  result = pd.DataFrame(rows)
  if result.empty:
    return pd.DataFrame(columns=columns)

  result["topic_display"] = result["topic"].map(_pretty_text)
  result["genre_display"] = result["genre"].map(_pretty_text)
  return result[columns]


def _build_keyword_driver_df(
  filtered_sentiment_keyword: pd.DataFrame,
  filtered_sentiment_comments: pd.DataFrame,
  positive: bool,
  top_n: int = 15,
) -> pd.DataFrame:
  if not filtered_sentiment_keyword.empty:
    gold_df = _build_keyword_driver_df_from_gold(
      filtered_sentiment_keyword=filtered_sentiment_keyword,
      positive=positive,
      top_n=top_n,
    )
    if not gold_df.empty:
      return gold_df

  return _build_keyword_driver_df_from_comments(
    filtered_sentiment_comments=filtered_sentiment_comments,
    positive=positive,
    top_n=top_n,
  )


def _build_daily_topic_breakdown(filtered_sentiment_comments: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_comments.empty:
    return pd.DataFrame()

  working = filtered_sentiment_comments.copy()

  if "collection_date" in working.columns:
    working["collection_date"] = pd.to_datetime(working["collection_date"], errors="coerce")
  elif "comment_published_at" in working.columns:
    working["collection_date"] = pd.to_datetime(working["comment_published_at"], errors="coerce", utc=True).dt.floor("D")
  else:
    return pd.DataFrame()

  if "topic" not in working.columns:
    working["topic"] = "unknown"
  if "genre" not in working.columns:
    working["genre"] = "unknown"
  if "comment_id" not in working.columns:
    working["comment_id"] = range(len(working))
  if "sentiment_score" not in working.columns:
    working["sentiment_score"] = 0.0
  if "sentiment_label" not in working.columns:
    working["sentiment_label"] = "neutral"

  working = working.dropna(subset=["collection_date"]).copy()

  grouped = (
    working.groupby(["collection_date", "topic", "genre"], as_index=False)
    .agg(
      comments_count=("comment_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
      negative_comments=("sentiment_label", lambda s: int((s.astype(str) == "negative").sum())),
      positive_comments=("sentiment_label", lambda s: int((s.astype(str) == "positive").sum())),
    )
  )

  denominator = grouped["comments_count"].replace(0, 1)
  grouped["negative_ratio"] = grouped["negative_comments"] / denominator
  grouped["positive_ratio"] = grouped["positive_comments"] / denominator
  grouped["topic_display"] = grouped["topic"].map(_pretty_text)
  grouped["genre_display"] = grouped["genre"].map(_pretty_text)

  return grouped


def _build_sentiment_endorsement_view(filtered_sentiment_comments: pd.DataFrame) -> pd.DataFrame:
  columns = [
    "sentiment_bucket",
    "sentiment_bucket_order",
    "comments_count",
    "total_comment_likes",
    "avg_likes_per_comment",
    "avg_sentiment_score",
  ]

  if filtered_sentiment_comments.empty:
    return pd.DataFrame(columns=columns)

  working = filtered_sentiment_comments.copy()

  if "sentiment_score" not in working.columns:
    return pd.DataFrame(columns=columns)

  like_col = _resolve_like_count_column(working)
  if like_col is None:
    working["__resolved_like_count"] = 0.0
  else:
    working["__resolved_like_count"] = pd.to_numeric(
      working[like_col],
      errors="coerce",
    ).fillna(0.0)

  working["sentiment_score"] = pd.to_numeric(
    working["sentiment_score"],
    errors="coerce",
  )

  working = working.dropna(subset=["sentiment_score"]).copy()
  if working.empty:
    return pd.DataFrame(columns=columns)

  bins = [-1.01, -0.35, -0.05, 0.05, 0.35, 1.01]
  labels = [
    "Very Negative",
    "Negative",
    "Neutral",
    "Positive",
    "Very Positive",
  ]

  working["sentiment_bucket"] = pd.cut(
    working["sentiment_score"],
    bins=bins,
    labels=labels,
    include_lowest=True,
  )

  grouped = (
    working.groupby("sentiment_bucket", observed=False, as_index=False)
    .agg(
      comments_count=("sentiment_bucket", "size"),
      total_comment_likes=("__resolved_like_count", "sum"),
      avg_likes_per_comment=("__resolved_like_count", "mean"),
      avg_sentiment_score=("sentiment_score", "mean"),
    )
  )

  bucket_order = {label: idx for idx, label in enumerate(labels)}
  grouped["sentiment_bucket"] = grouped["sentiment_bucket"].astype(str)
  grouped["sentiment_bucket_order"] = grouped["sentiment_bucket"].map(bucket_order)
  grouped = grouped.sort_values("sentiment_bucket_order")

  return grouped[columns]


def _build_like_distribution_by_label(filtered_sentiment_comments: pd.DataFrame) -> pd.DataFrame:
  columns = ["sentiment_label", "resolved_like_count"]

  if filtered_sentiment_comments.empty:
    return pd.DataFrame(columns=columns)

  working = filtered_sentiment_comments.copy()

  if "sentiment_label" not in working.columns:
    return pd.DataFrame(columns=columns)

  like_col = _resolve_like_count_column(working)
  if like_col is None:
    working["resolved_like_count"] = 0.0
  else:
    working["resolved_like_count"] = pd.to_numeric(
      working[like_col],
      errors="coerce",
    ).fillna(0.0)

  working["sentiment_label"] = (
    working["sentiment_label"]
    .fillna("neutral")
    .astype(str)
    .str.strip()
    .str.lower()
  )

  allowed = {"positive", "neutral", "negative"}
  working = working[working["sentiment_label"].isin(allowed)].copy()
  if working.empty:
    return pd.DataFrame(columns=columns)

  return working[columns]


def render_diagnostic_tab(
  filtered_sentiment_topic_summary: pd.DataFrame,
  filtered_sentiment_video_summary: pd.DataFrame,
  filtered_sentiment_keyword: pd.DataFrame,
  filtered_sentiment_daily_trend: pd.DataFrame,
  filtered_sentiment_comments: pd.DataFrame,
  format_score: Callable[[float | int], str],
  format_pct: Callable[[float | int], str],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Diagnostic Analytics")
  st.caption(
    "This page explains why sentiment is moving up or down by identifying concentrated risk topics, worst-day drivers, video-level sentiment pressure, keyword-level driver words, and which kinds of comments are actually getting audience endorsement through likes."
  )

  if (
    filtered_sentiment_topic_summary.empty
    and filtered_sentiment_video_summary.empty
    and filtered_sentiment_comments.empty
  ):
    show_empty_state("No diagnostic sentiment data is available for the selected filters.")
    return

  topic_summary = (
    _build_topic_summary_from_comments(filtered_sentiment_comments)
    if not filtered_sentiment_comments.empty
    else filtered_sentiment_topic_summary.copy()
  )

  video_summary = (
    _build_video_summary_from_comments(filtered_sentiment_comments)
    if not filtered_sentiment_comments.empty
    else filtered_sentiment_video_summary.copy()
  )

  negative_keywords = _build_keyword_driver_df(
    filtered_sentiment_keyword=filtered_sentiment_keyword,
    filtered_sentiment_comments=filtered_sentiment_comments,
    positive=False,
    top_n=15,
  )
  positive_keywords = _build_keyword_driver_df(
    filtered_sentiment_keyword=filtered_sentiment_keyword,
    filtered_sentiment_comments=filtered_sentiment_comments,
    positive=True,
    top_n=15,
  )

  daily_topic_breakdown = _build_daily_topic_breakdown(filtered_sentiment_comments)
  endorsement_view = _build_sentiment_endorsement_view(filtered_sentiment_comments)
  like_distribution_view = _build_like_distribution_by_label(filtered_sentiment_comments)

  st.markdown("### Key Diagnostic Insights")

  insights: list[str] = []
  worst_day_for_breakdown: pd.Timestamp | None = None

  if not topic_summary.empty:
    at_risk_topic = topic_summary.sort_values(
      ["negative_ratio", "comments_count"],
      ascending=[False, False],
    ).iloc[0]

    strongest_topic = topic_summary.sort_values(
      ["avg_sentiment_score", "comments_count"],
      ascending=[False, False],
    ).iloc[0]

    insights.append(
      f"The most at-risk topic is **{at_risk_topic['topic_display']}** in **{at_risk_topic['genre_display']}**, with negative share {format_pct(at_risk_topic.get('negative_ratio', 0.0))} and average sentiment {format_score(at_risk_topic.get('avg_sentiment_score', 0.0))}."
    )

    insights.append(
      f"The strongest topic is **{strongest_topic['topic_display']}** in **{strongest_topic['genre_display']}**, with average sentiment {format_score(strongest_topic.get('avg_sentiment_score', 0.0))}."
    )

  if not filtered_sentiment_daily_trend.empty:
    trend_df = filtered_sentiment_daily_trend.copy()
    trend_df["collection_date"] = pd.to_datetime(trend_df["collection_date"], errors="coerce")
    trend_df = trend_df.dropna(subset=["collection_date"]).sort_values("collection_date")

    if "avg_sentiment_score" in trend_df.columns and len(trend_df) >= 2:
      work = (
        trend_df.groupby("collection_date", as_index=False)
        .agg(avg_sentiment_score=("avg_sentiment_score", "mean"))
        .sort_values("collection_date")
      )
      work["delta"] = work["avg_sentiment_score"].diff()

      falling = work.dropna(subset=["delta"]).sort_values("delta", ascending=True)
      if not falling.empty:
        worst_day = falling.iloc[0]
        worst_day_for_breakdown = worst_day["collection_date"]
        insights.append(
          f"The sharpest one-day sentiment drop happened on **{worst_day_for_breakdown.strftime('%Y-%m-%d')}**, with a change of {format_score(worst_day['delta'])}."
        )

  if not endorsement_view.empty:
    most_endorsed_bucket = endorsement_view.sort_values(
      ["avg_likes_per_comment", "total_comment_likes"],
      ascending=[False, False],
    ).iloc[0]
    insights.append(
      f"The strongest audience endorsement currently comes from **{most_endorsed_bucket['sentiment_bucket']}** comments, with average likes per comment {format_score(most_endorsed_bucket['avg_likes_per_comment'])}."
    )

  for insight in insights:
    st.markdown(f"- {insight}")

  if not topic_summary.empty:
    st.markdown("### Root-Cause Overview")
    root_left, root_right = st.columns(2)

    with root_left:
      st.markdown("#### Topic Risk Map")
      st.caption(
        "Topics higher on the chart have stronger negative share. Topics further left have weaker sentiment. Bigger bubbles mean the issue affects more discussion volume."
      )

      risk_df = topic_summary.copy()
      fig_risk_map = px.scatter(
        risk_df,
        x="avg_sentiment_score",
        y="negative_ratio",
        size="comments_count",
        color="genre_display",
        hover_name="topic_display",
        hover_data={
          "genre_display": True,
          "comments_count": ":,.0f",
          "videos_covered": ":,.0f" if "videos_covered" in risk_df.columns else False,
          "avg_comment_likes": ":.2f" if "avg_comment_likes" in risk_df.columns else False,
          "avg_sentiment_score": ":.3f",
          "negative_ratio": ":.3f",
        },
        labels={
          "avg_sentiment_score": "Average Sentiment Score",
          "negative_ratio": "Negative Share",
          "comments_count": "Matched Comments",
          "genre_display": "Genre",
          "topic_display": "Topic",
          "videos_covered": "Videos Covered",
          "avg_comment_likes": "Average Comment Likes",
        },
        template="plotly_dark",
        title="Topic Risk Map",
        size_max=42,
      )
      fig_risk_map.update_layout(
        xaxis_title="Average Sentiment Score",
        yaxis_title="Negative Share",
        legend_title="Genre",
        height=460,
      )
      st.plotly_chart(fig_risk_map, use_container_width=True)

    with root_right:
      st.markdown("#### Negative Comment Load by Topic")
      st.caption(
        "This chart shows where weak sentiment is most concentrated by combining total topic discussion with the count of negative comments."
      )

      contribution_df = (
        topic_summary[["topic_display", "genre_display", "negative_comments", "comments_count", "avg_sentiment_score"]]
        .copy()
        .sort_values(["negative_comments", "comments_count"], ascending=[False, False])
        .head(10)
      )

      if contribution_df.empty:
        st.info("Topic contribution data is not available for this view.")
      else:
        fig_contribution = px.bar(
          contribution_df.sort_values("negative_comments", ascending=True),
          x="negative_comments",
          y="topic_display",
          color="genre_display",
          orientation="h",
          hover_data={
            "comments_count": ":,.0f",
            "avg_sentiment_score": ":.3f",
            "negative_comments": ":,.0f",
            "genre_display": False,
          },
          labels={
            "negative_comments": "Negative Comments",
            "topic_display": "Topic",
            "comments_count": "Matched Comments",
            "avg_sentiment_score": "Average Sentiment Score",
            "genre_display": "Genre",
          },
          template="plotly_dark",
          title="Negative Comment Load by Topic",
        )
        fig_contribution.update_layout(
          xaxis_title="Negative Comments",
          yaxis_title="Topic",
          legend_title="Genre",
          height=460,
        )
        st.plotly_chart(fig_contribution, use_container_width=True)

  if worst_day_for_breakdown is not None and not daily_topic_breakdown.empty:
    st.markdown("### Drivers of the Sharpest One-Day Drop")
    st.caption(
      f"This section isolates the topic-level breakdown for **{worst_day_for_breakdown.strftime('%Y-%m-%d')}**, the day with the weakest one-day sentiment change in the current filtered view."
    )

    worst_day_topics = (
      daily_topic_breakdown[daily_topic_breakdown["collection_date"] == worst_day_for_breakdown]
      .copy()
      .sort_values(["negative_comments", "comments_count"], ascending=[False, False])
      .head(10)
    )

    if worst_day_topics.empty:
      st.info("Worst-day topic decomposition is not available for this view.")
    else:
      fig_worst_day = px.bar(
        worst_day_topics.sort_values("negative_comments", ascending=True),
        x="negative_comments",
        y="topic_display",
        color="genre_display",
        orientation="h",
        hover_data={
          "comments_count": ":,.0f",
          "avg_sentiment_score": ":.3f",
          "negative_ratio": ":.3f",
          "genre_display": False,
        },
        labels={
          "negative_comments": "Negative Comments",
          "topic_display": "Topic",
          "comments_count": "Matched Comments",
          "avg_sentiment_score": "Average Sentiment Score",
          "negative_ratio": "Negative Share",
          "genre_display": "Genre",
        },
        template="plotly_dark",
        title="Topics Behind the Sharpest One-Day Drop",
      )
      fig_worst_day.update_layout(
        xaxis_title="Negative Comments",
        yaxis_title="Topic",
        legend_title="Genre",
        height=460,
      )
      st.plotly_chart(fig_worst_day, use_container_width=True)

  if not video_summary.empty:
    st.markdown("### Video-Level Sentiment Drivers")
    st.caption(
      "These charts show which specific videos are dragging sentiment down and which ones are lifting it up in the current filtered view."
    )

    weak_videos = (
      video_summary
      .sort_values(["avg_sentiment_score", "comments_count", "total_comment_likes"], ascending=[True, False, False])
      .head(10)
      .copy()
    )

    strong_videos = (
      video_summary
      .sort_values(["avg_sentiment_score", "comments_count", "total_comment_likes"], ascending=[False, False, False])
      .head(10)
      .copy()
    )

    left_videos, right_videos = st.columns(2)

    with left_videos:
      fig_weak_videos = px.bar(
        weak_videos.sort_values("avg_sentiment_score", ascending=False),
        x="avg_sentiment_score",
        y="video_display",
        color="genre_display",
        orientation="h",
        hover_name="video_title",
        hover_data={
          "channel_title": True,
          "topic_display": True,
          "comments_count": ":,.0f",
          "negative_ratio": ":.3f",
          "total_comment_likes": ":,.0f",
          "avg_sentiment_score": ":.3f",
          "genre_display": False,
          "video_display": False,
        },
        labels={
          "avg_sentiment_score": "Average Sentiment Score",
          "video_display": "Video",
          "topic_display": "Topic",
          "comments_count": "Matched Comments",
          "negative_ratio": "Negative Share",
          "total_comment_likes": "Total Comment Likes",
          "channel_title": "Channel",
          "genre_display": "Genre",
        },
        template="plotly_dark",
        title="Videos Pulling Sentiment Down",
      )
      fig_weak_videos.update_layout(
        xaxis_title="Average Sentiment Score",
        yaxis_title="Video",
        legend_title="Genre",
        height=500,
      )
      st.plotly_chart(fig_weak_videos, use_container_width=True)

    with right_videos:
      fig_strong_videos = px.bar(
        strong_videos.sort_values("avg_sentiment_score", ascending=True),
        x="avg_sentiment_score",
        y="video_display",
        color="genre_display",
        orientation="h",
        hover_name="video_title",
        hover_data={
          "channel_title": True,
          "topic_display": True,
          "comments_count": ":,.0f",
          "positive_ratio": ":.3f",
          "total_comment_likes": ":,.0f",
          "avg_sentiment_score": ":.3f",
          "genre_display": False,
          "video_display": False,
        },
        labels={
          "avg_sentiment_score": "Average Sentiment Score",
          "video_display": "Video",
          "topic_display": "Topic",
          "comments_count": "Matched Comments",
          "positive_ratio": "Positive Share",
          "total_comment_likes": "Total Comment Likes",
          "channel_title": "Channel",
          "genre_display": "Genre",
        },
        template="plotly_dark",
        title="Videos Lifting Sentiment Up",
      )
      fig_strong_videos.update_layout(
        xaxis_title="Average Sentiment Score",
        yaxis_title="Video",
        legend_title="Genre",
        height=500,
      )
      st.plotly_chart(fig_strong_videos, use_container_width=True)

  st.markdown("### Keyword-Level Sentiment Drivers")
  st.caption(
    "These driver words help explain what kinds of language are most associated with negative or positive audience response in the current view."
  )

  left_col, right_col = st.columns(2)

  with left_col:
    if negative_keywords.empty:
      st.info("No negative driver keywords are available for this view.")
    else:
      fig_negative = px.bar(
        negative_keywords.sort_values("keyword_count", ascending=True),
        x="keyword_count",
        y="keyword",
        color="topic_display",
        orientation="h",
        hover_data={
          "genre_display": True,
          "distinct_comments": ":,.0f",
          "avg_sentiment_score": ":.3f",
          "keyword_count": ":,.0f",
          "topic_display": False,
        },
        labels={
          "keyword_count": "Keyword Frequency",
          "keyword": "Keyword",
          "distinct_comments": "Distinct Comments",
          "avg_sentiment_score": "Average Sentiment Score",
          "genre_display": "Genre",
          "topic_display": "Topic",
        },
        template="plotly_dark",
        title="Frequent Negative Driver Keywords",
      )
      fig_negative.update_layout(
        xaxis_title="Keyword Frequency",
        yaxis_title="Keyword",
        legend_title="Topic",
        height=470,
      )
      st.plotly_chart(fig_negative, use_container_width=True)

  with right_col:
    if positive_keywords.empty:
      st.info("No positive driver keywords are available for this view.")
    else:
      fig_positive = px.bar(
        positive_keywords.sort_values("keyword_count", ascending=True),
        x="keyword_count",
        y="keyword",
        color="topic_display",
        orientation="h",
        hover_data={
          "genre_display": True,
          "distinct_comments": ":,.0f",
          "avg_sentiment_score": ":.3f",
          "keyword_count": ":,.0f",
          "topic_display": False,
        },
        labels={
          "keyword_count": "Keyword Frequency",
          "keyword": "Keyword",
          "distinct_comments": "Distinct Comments",
          "avg_sentiment_score": "Average Sentiment Score",
          "genre_display": "Genre",
          "topic_display": "Topic",
        },
        template="plotly_dark",
        title="Frequent Positive Driver Keywords",
      )
      fig_positive.update_layout(
        xaxis_title="Keyword Frequency",
        yaxis_title="Keyword",
        legend_title="Topic",
        height=470,
      )
      st.plotly_chart(fig_positive, use_container_width=True)

  st.markdown("### Audience Endorsement Diagnostics")
  st.caption(
    "These visuals show which kinds of comments are actually getting audience support through likes. This is more useful here than a generic correlation matrix because it directly answers what type of sentiment gets endorsed."
  )

  endorse_left, endorse_right = st.columns(2)

  with endorse_left:
    if endorsement_view.empty:
      st.info("Not enough data is available to show endorsement by sentiment bucket.")
    else:
      fig_endorsement = px.bar(
        endorsement_view,
        x="sentiment_bucket",
        y="avg_likes_per_comment",
        color="sentiment_bucket",
        hover_data={
          "comments_count": ":,.0f",
          "total_comment_likes": ":,.0f",
          "avg_sentiment_score": ":.3f",
          "avg_likes_per_comment": ":.2f",
          "sentiment_bucket_order": False,
        },
        labels={
          "sentiment_bucket": "Sentiment Bucket",
          "avg_likes_per_comment": "Average Likes per Comment",
          "comments_count": "Matched Comments",
          "total_comment_likes": "Total Comment Likes",
          "avg_sentiment_score": "Average Sentiment Score",
        },
        template="plotly_dark",
        title="Audience Endorsement by Sentiment Bucket",
      )
      fig_endorsement.update_layout(
        xaxis_title="Sentiment Bucket",
        yaxis_title="Average Likes per Comment",
        showlegend=False,
        height=470,
      )
      st.plotly_chart(fig_endorsement, use_container_width=True)

  with endorse_right:
    if like_distribution_view.empty:
      st.info("Not enough data is available to show like distribution by sentiment label.")
    else:
      fig_like_dist = px.box(
        like_distribution_view,
        x="sentiment_label",
        y="resolved_like_count",
        color="sentiment_label",
        points="outliers",
        labels={
          "sentiment_label": "Sentiment Label",
          "resolved_like_count": "Comment Likes",
        },
        template="plotly_dark",
        title="Like Distribution by Sentiment Label",
      )
      fig_like_dist.update_layout(
        xaxis_title="Sentiment Label",
        yaxis_title="Comment Likes",
        showlegend=False,
        height=470,
      )
      st.plotly_chart(fig_like_dist, use_container_width=True)