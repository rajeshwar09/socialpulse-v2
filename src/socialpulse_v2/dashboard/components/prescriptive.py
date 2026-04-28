from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def _pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


def _find_negative_keywords(
  filtered_sentiment_keyword: pd.DataFrame,
  topic: str,
  limit: int = 3,
) -> list[str]:
  if filtered_sentiment_keyword.empty or "topic" not in filtered_sentiment_keyword.columns:
    return []

  keyword_count_col = None
  for candidate in ["keyword_count", "keyword_frequency", "occurrences", "mentions_count", "count"]:
    if candidate in filtered_sentiment_keyword.columns:
      keyword_count_col = candidate
      break

  if keyword_count_col is None:
    return []

  subset = filtered_sentiment_keyword[
    (filtered_sentiment_keyword["topic"].astype(str) == str(topic))
    & (filtered_sentiment_keyword["avg_sentiment_score"] < 0)
  ].copy()

  if subset.empty:
    return []

  subset = subset.sort_values([keyword_count_col, "avg_sentiment_score"], ascending=[False, True])
  return subset["keyword"].astype(str).head(limit).tolist()


def _build_topic_summary_from_comments(filtered_sentiment_comments: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_comments.empty:
    return pd.DataFrame()

  work = filtered_sentiment_comments.copy()

  if "comment_id" not in work.columns:
    work["comment_id"] = range(len(work))
  if "video_id" not in work.columns:
    work["video_id"] = ""
  if "topic" not in work.columns:
    work["topic"] = "unknown"
  if "genre" not in work.columns:
    work["genre"] = "unknown"
  if "sentiment_score" not in work.columns:
    work["sentiment_score"] = 0.0
  if "sentiment_label" not in work.columns:
    work["sentiment_label"] = "neutral"

  out = (
    work.groupby(["topic", "genre"], as_index=False)
    .agg(
      comments_count=("comment_id", "nunique"),
      videos_covered=("video_id", "nunique"),
      avg_sentiment_score=("sentiment_score", "mean"),
      positive_comments=("sentiment_label", lambda s: int((s.astype(str) == "positive").sum())),
      neutral_comments=("sentiment_label", lambda s: int((s.astype(str) == "neutral").sum())),
      negative_comments=("sentiment_label", lambda s: int((s.astype(str) == "negative").sum())),
    )
    .sort_values(["comments_count", "avg_sentiment_score"], ascending=[False, False])
  )

  denominator = out["comments_count"].replace(0, 1)
  out["positive_ratio"] = out["positive_comments"] / denominator
  out["negative_ratio"] = out["negative_comments"] / denominator

  return out


def render_prescriptive_tab(
  filtered_sentiment_topic_summary: pd.DataFrame,
  filtered_sentiment_keyword: pd.DataFrame,
  filtered_sentiment_comments: pd.DataFrame,
  format_score: Callable[[float | int], str],
  format_pct: Callable[[float | int], str],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Prescriptive Analytics")
  st.caption(
    "These recommendations are based on comment sentiment patterns and topic-level audience response."
  )

  working_df = (
    filtered_sentiment_topic_summary.copy()
    if not filtered_sentiment_topic_summary.empty
    else _build_topic_summary_from_comments(filtered_sentiment_comments)
  )

  if working_df.empty:
    show_empty_state("No prescriptive sentiment data is available for the selected filters.")
    return

  if "comments_count" not in working_df.columns:
    working_df["comments_count"] = 0
  if "avg_sentiment_score" not in working_df.columns:
    working_df["avg_sentiment_score"] = 0.0
  if "positive_ratio" not in working_df.columns:
    working_df["positive_ratio"] = 0.0
  if "negative_ratio" not in working_df.columns:
    working_df["negative_ratio"] = 0.0

  total_comments = max(float(working_df["comments_count"].sum()), 1.0)
  working_df["volume_share"] = working_df["comments_count"] / total_comments
  working_df["sentiment_risk"] = (1 - working_df["avg_sentiment_score"].clip(-1, 1)) / 2
  working_df["monitoring_priority"] = (
    0.50 * working_df["negative_ratio"]
    + 0.30 * working_df["sentiment_risk"]
    + 0.20 * working_df["volume_share"]
  )

  ranked_risk = working_df.sort_values(
    ["monitoring_priority", "comments_count"],
    ascending=[False, False],
  )

  ranked_growth = working_df.sort_values(
    ["positive_ratio", "avg_sentiment_score", "comments_count"],
    ascending=[False, False, False],
  )

  recommendations: list[str] = []
  action_rows: list[dict[str, str]] = []

  for _, row in ranked_risk.head(3).iterrows():
    topic = _pretty_text(row["topic"])
    genre = _pretty_text(row["genre"])
    negative_keywords = _find_negative_keywords(filtered_sentiment_keyword, str(row["topic"]))
    keyword_text = f" Frequent negative words include: {', '.join(negative_keywords)}." if negative_keywords else ""

    recommendations.append(
      f"Monitor Risk — Topic '{topic}' in genre '{genre}' needs close tracking. Negative share is {format_pct(row['negative_ratio'])} and average sentiment is {format_score(row['avg_sentiment_score'])}.{keyword_text}"
    )
    action_rows.append({"action": "Monitor", "topic": topic})

  for _, row in ranked_growth.head(3).iterrows():
    topic = _pretty_text(row["topic"])
    genre = _pretty_text(row["genre"])

    recommendations.append(
      f"Growth Opportunity — Topic '{topic}' in genre '{genre}' is performing well. Positive share is {format_pct(row['positive_ratio'])} with average sentiment {format_score(row['avg_sentiment_score'])}. Consider expanding collection in this area."
    )
    action_rows.append({"action": "Growth Opportunity", "topic": topic})

  st.markdown("### Recommended Actions")
  for index, recommendation in enumerate(recommendations, start=1):
    st.markdown(
      f"""
<div style="padding:12px 14px; border:1px solid rgba(255,255,255,0.08); border-radius:12px; margin-bottom:10px; background:rgba(255,255,255,0.02);">
  <b>Recommendation {index}.</b> {recommendation}
</div>
""",
      unsafe_allow_html=True,
    )

  st.markdown("### Monitoring Priority by Topic")

  priority_df = ranked_risk.head(10).copy()
  priority_df["topic_display"] = priority_df["topic"].map(_pretty_text)
  priority_df["genre_display"] = priority_df["genre"].map(_pretty_text)

  fig_priority = px.bar(
    priority_df,
    x="monitoring_priority",
    y="topic_display",
    color="genre_display",
    orientation="h",
    template="plotly_dark",
    title="Monitoring Priority by Topic",
  )
  fig_priority.update_layout(
    xaxis_title="Priority Score",
    yaxis_title="Topic",
    legend_title="Genre",
    height=480,
    yaxis={"categoryorder": "total ascending"},
  )
  st.plotly_chart(fig_priority, use_container_width=True)

  if action_rows:
    st.markdown("### Recommended Action Mix")

    action_df = pd.DataFrame(action_rows)
    mix_df = action_df.groupby("action", as_index=False).size().rename(columns={"size": "count"})

    fig_mix = px.pie(
      mix_df,
      values="count",
      names="action",
      template="plotly_dark",
      title="Recommended Action Mix",
    )
    fig_mix.update_layout(height=420)
    st.plotly_chart(fig_mix, use_container_width=True)