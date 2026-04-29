from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def _pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


def _find_keywords(
  filtered_sentiment_keyword: pd.DataFrame,
  topic: str,
  polarity: str,
  limit: int = 3,
) -> list[str]:
  if filtered_sentiment_keyword.empty:
    return []

  if "topic" not in filtered_sentiment_keyword.columns or "keyword" not in filtered_sentiment_keyword.columns:
    return []

  keyword_count_col = None
  for candidate in ["keyword_count", "keyword_frequency", "occurrences", "mentions_count", "count"]:
    if candidate in filtered_sentiment_keyword.columns:
      keyword_count_col = candidate
      break

  if keyword_count_col is None or "avg_sentiment_score" not in filtered_sentiment_keyword.columns:
    return []

  subset = filtered_sentiment_keyword[
    filtered_sentiment_keyword["topic"].astype(str) == str(topic)
  ].copy()

  if subset.empty:
    return []

  if polarity == "negative":
    subset = subset[subset["avg_sentiment_score"] < 0]
    subset = subset.sort_values([keyword_count_col, "avg_sentiment_score"], ascending=[False, True])
  else:
    subset = subset[subset["avg_sentiment_score"] > 0]
    subset = subset.sort_values([keyword_count_col, "avg_sentiment_score"], ascending=[False, False])

  if subset.empty:
    return []

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


def _render_metric_chip(label: str, value: str) -> str:
  return (
    "<span style=\"display:inline-block; margin:4px 8px 4px 0; padding:6px 10px; "
    "border:1px solid rgba(255,255,255,0.08); border-radius:999px; "
    "background:rgba(255,255,255,0.03); font-size:0.92rem;\">"
    f"<b>{label}:</b> {value}</span>"
  )


def _render_recommendation_card(
  action_label: str,
  topic: str,
  genre: str,
  comments_count: int,
  videos_covered: int,
  avg_sentiment_score: float,
  positive_ratio: float,
  negative_ratio: float,
  driver_words: list[str],
  action_text: str,
  priority_score: float,
) -> None:
  action_color = "#ff6b6b" if action_label == "Monitor Risk" else "#3ddc97"
  words_text = ", ".join(driver_words) if driver_words else "No strong driver words found yet"

  chips_html = "".join(
    [
      _render_metric_chip("Comments", f"{comments_count:,}"),
      _render_metric_chip("Videos", f"{videos_covered:,}"),
      _render_metric_chip("Avg Sentiment", f"{avg_sentiment_score:.3f}"),
      _render_metric_chip("Positive Share", f"{positive_ratio * 100:.1f}%"),
      _render_metric_chip("Negative Share", f"{negative_ratio * 100:.1f}%"),
      _render_metric_chip("Priority Score", f"{priority_score:.3f}"),
    ]
  )

  st.markdown(
    f"""
<div style="padding:14px 16px; border:1px solid rgba(255,255,255,0.08); border-radius:14px; margin-bottom:12px; background:rgba(255,255,255,0.02);">
  <div style="display:inline-block; margin-bottom:10px; padding:5px 10px; border-radius:999px; background:{action_color}; color:#06111f; font-weight:700; font-size:0.9rem;">
    {action_label}
  </div>
  <div style="font-size:1.08rem; font-weight:700; margin-bottom:8px;">
    {_pretty_text(topic)} <span style="opacity:0.8; font-weight:500;">· {_pretty_text(genre)}</span>
  </div>
  <div style="margin-bottom:8px;">{chips_html}</div>
  <div style="margin-bottom:6px;"><b>Why this topic:</b> Driver words suggest audience reaction is linked to: {words_text}</div>
  <div><b>Recommended action:</b> {action_text}</div>
</div>
""",
    unsafe_allow_html=True,
  )


def _render_reason_card(
  title: str,
  topic: str,
  genre: str,
  explanation: str,
) -> None:
  st.markdown(
    f"""
<div style="padding:14px 16px; border:1px solid rgba(255,255,255,0.08); border-radius:14px; margin-bottom:12px; background:rgba(255,255,255,0.02); height:100%;">
  <div style="font-size:1rem; font-weight:700; margin-bottom:8px;">{title}</div>
  <div style="margin-bottom:6px;"><b>{_pretty_text(topic)}</b> · {_pretty_text(genre)}</div>
  <div style="opacity:0.92;">{explanation}</div>
</div>
""",
    unsafe_allow_html=True,
  )


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
    "This page turns audience sentiment patterns into action suggestions, so you can quickly see what to monitor, what to expand, and why each recommendation matters."
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
  if "videos_covered" not in working_df.columns:
    working_df["videos_covered"] = 0
  if "avg_sentiment_score" not in working_df.columns:
    working_df["avg_sentiment_score"] = 0.0
  if "positive_ratio" not in working_df.columns:
    working_df["positive_ratio"] = 0.0
  if "negative_ratio" not in working_df.columns:
    working_df["negative_ratio"] = 0.0
  if "topic" not in working_df.columns:
    working_df["topic"] = "unknown"
  if "genre" not in working_df.columns:
    working_df["genre"] = "unknown"

  total_comments = max(float(working_df["comments_count"].sum()), 1.0)
  total_videos = max(float(working_df["videos_covered"].sum()), 1.0)

  working_df["volume_share"] = working_df["comments_count"] / total_comments
  working_df["video_share"] = working_df["videos_covered"] / total_videos
  working_df["sentiment_risk"] = (1 - working_df["avg_sentiment_score"].clip(-1, 1)) / 2
  working_df["sentiment_strength"] = (working_df["avg_sentiment_score"].clip(-1, 1) + 1) / 2

  working_df["monitoring_priority"] = (
    0.45 * working_df["negative_ratio"]
    + 0.25 * working_df["sentiment_risk"]
    + 0.20 * working_df["volume_share"]
    + 0.10 * (1 - working_df["positive_ratio"])
  )

  working_df["growth_priority"] = (
    0.40 * working_df["positive_ratio"]
    + 0.25 * working_df["sentiment_strength"]
    + 0.20 * working_df["volume_share"]
    + 0.15 * working_df["video_share"]
  )

  ranked_risk = working_df.sort_values(
    ["monitoring_priority", "comments_count"],
    ascending=[False, False],
  ).copy()

  ranked_growth = working_df.sort_values(
    ["growth_priority", "comments_count"],
    ascending=[False, False],
  ).copy()

  risk_rows = ranked_risk.head(3).copy()
  growth_rows = ranked_growth.head(3).copy()

  risk_topic = _pretty_text(risk_rows.iloc[0]["topic"]) if not risk_rows.empty else "N/A"
  growth_topic = _pretty_text(growth_rows.iloc[0]["topic"]) if not growth_rows.empty else "N/A"
  risk_genre = _pretty_text(risk_rows.iloc[0]["genre"]) if not risk_rows.empty else "N/A"
  growth_genre = _pretty_text(growth_rows.iloc[0]["genre"]) if not growth_rows.empty else "N/A"

  st.markdown("### Executive Summary")
  st.markdown(
    f"""
<div style="padding:14px 16px; border:1px solid rgba(255,255,255,0.08); border-radius:14px; margin-bottom:14px; background:rgba(255,255,255,0.02);">
  The current filtered audience view shows <b>{len(risk_rows)}</b> topics that need close monitoring and <b>{len(growth_rows)}</b> topics that look suitable for expansion.
  The strongest immediate risk is <b>{risk_topic}</b> in <b>{risk_genre}</b>, while the strongest growth opportunity is <b>{growth_topic}</b> in <b>{growth_genre}</b>.
</div>
""",
    unsafe_allow_html=True,
  )

  k1, k2, k3, k4 = st.columns(4)
  k1.metric("Topics to Monitor", f"{len(risk_rows)}")
  k2.metric("Growth Opportunities", f"{len(growth_rows)}")
  k3.metric("Top Risk Topic", risk_topic)
  k4.metric("Top Opportunity Topic", growth_topic)

  st.markdown("### Recommended Actions")

  for _, row in risk_rows.iterrows():
    topic = str(row["topic"])
    genre = str(row["genre"])
    negative_words = _find_keywords(filtered_sentiment_keyword, topic, polarity="negative", limit=3)

    action_text = (
      "Track this topic more closely, review recent videos and comments, and investigate the negative driver words before expanding coverage."
    )

    _render_recommendation_card(
      action_label="Monitor Risk",
      topic=topic,
      genre=genre,
      comments_count=int(row["comments_count"]),
      videos_covered=int(row["videos_covered"]),
      avg_sentiment_score=float(row["avg_sentiment_score"]),
      positive_ratio=float(row["positive_ratio"]),
      negative_ratio=float(row["negative_ratio"]),
      driver_words=negative_words,
      action_text=action_text,
      priority_score=float(row["monitoring_priority"]),
    )

  for _, row in growth_rows.iterrows():
    topic = str(row["topic"])
    genre = str(row["genre"])
    positive_words = _find_keywords(filtered_sentiment_keyword, topic, polarity="positive", limit=3)

    driver_text = (
      "Positive driver words suggest this topic is connecting well with viewers."
      if positive_words else
      "Audience response is positive enough to justify more collection or deeper monitoring."
    )

    action_text = (
      "Consider expanding collection in this area, increase query coverage, and use this topic as a benchmark for stronger audience response."
    )

    _render_recommendation_card(
      action_label="Growth Opportunity",
      topic=topic,
      genre=genre,
      comments_count=int(row["comments_count"]),
      videos_covered=int(row["videos_covered"]),
      avg_sentiment_score=float(row["avg_sentiment_score"]),
      positive_ratio=float(row["positive_ratio"]),
      negative_ratio=float(row["negative_ratio"]),
      driver_words=positive_words,
      action_text=f"{driver_text} {action_text}",
      priority_score=float(row["growth_priority"]),
    )

  st.markdown("### Monitoring Priority by Topic")
  st.caption(
    "Higher priority score means the topic combines weaker sentiment, stronger negative share, and enough discussion volume to influence the overall audience mood more heavily."
  )

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
    height=500,
    yaxis={"categoryorder": "total ascending"},
  )
  st.plotly_chart(fig_priority, use_container_width=True)

  st.markdown("### Why These Recommendations Were Chosen")
  st.caption(
    "These signals explain the decision logic behind the actions above, so the recommendation does not rely on only one chart or one score."
  )

  highest_negative_share = ranked_risk.sort_values(
    ["negative_ratio", "comments_count"],
    ascending=[False, False],
  ).iloc[0]

  largest_volume = working_df.sort_values(
    ["comments_count", "videos_covered"],
    ascending=[False, False],
  ).iloc[0]

  strongest_positive = ranked_growth.sort_values(
    ["positive_ratio", "comments_count"],
    ascending=[False, False],
  ).iloc[0]

  widest_coverage = working_df.sort_values(
    ["videos_covered", "comments_count"],
    ascending=[False, False],
  ).iloc[0]

  col1, col2 = st.columns(2)
  with col1:
    _render_reason_card(
      "Highest Negative Share",
      str(highest_negative_share["topic"]),
      str(highest_negative_share["genre"]),
      (
        f"This topic has the strongest negative reaction share in the current view "
        f"at {format_pct(highest_negative_share['negative_ratio'])}, so it is more likely to pull the overall mood down."
      ),
    )
    _render_reason_card(
      "Largest Discussion Volume",
      str(largest_volume["topic"]),
      str(largest_volume["genre"]),
      (
        f"This topic has the biggest audience discussion volume with {int(largest_volume['comments_count']):,} matched comments, "
        "so any sentiment change here can affect the dashboard story more strongly."
      ),
    )

  with col2:
    _render_reason_card(
      "Strongest Positive Opportunity",
      str(strongest_positive["topic"]),
      str(strongest_positive["genre"]),
      (
        f"This topic has the best positive audience response in the current view, with positive share "
        f"{format_pct(strongest_positive['positive_ratio'])} and average sentiment {format_score(strongest_positive['avg_sentiment_score'])}."
      ),
    )
    _render_reason_card(
      "Widest Video Coverage",
      str(widest_coverage["topic"]),
      str(widest_coverage["genre"]),
      (
        f"This topic already spans {int(widest_coverage['videos_covered']):,} videos, so it is a strong candidate for broader monitoring or benchmark comparison."
      ),
    )

  st.markdown("### Top 3 Next Moves")

  top_moves: list[str] = []
  if not risk_rows.empty:
    top_moves.append(
      f"Review the audience reaction for **{_pretty_text(risk_rows.iloc[0]['topic'])}** in **{_pretty_text(risk_rows.iloc[0]['genre'])}** first, because it currently has the highest monitoring priority."
    )
  if len(growth_rows) > 0:
    top_moves.append(
      f"Expand query coverage around **{_pretty_text(growth_rows.iloc[0]['topic'])}** in **{_pretty_text(growth_rows.iloc[0]['genre'])}**, because it is the strongest opportunity in the current view."
    )
  if not filtered_sentiment_keyword.empty:
    top_moves.append(
      "Use the driver words shown in the recommendation cards to explain *why* sentiment is weak or strong, instead of relying only on the raw score."
    )

  for move in top_moves[:3]:
    st.markdown(f"- {move}")