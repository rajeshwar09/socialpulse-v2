from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def resolve_keyword_count_column(df: pd.DataFrame) -> str | None:
  candidates = [
    "keyword_count",
    "keyword_frequency",
    "occurrences",
    "mentions_count",
    "count",
  ]
  for column in candidates:
    if column in df.columns:
      return column
  return None

def resolve_like_count_column(df: pd.DataFrame) -> str | None:
  candidates = [
    "like_count",
    "comment_like_count",
  ]
  for column in candidates:
    if column in df.columns:
      return column
  return None


def _pretty_text(value: object) -> str:
  return str(value).replace("_", " ").title()


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


def _truncate_text(text: object, limit: int = 42) -> str:
  value = str(text)
  if len(value) <= limit:
    return value
  return value[:limit].rstrip() + "..."


def _build_daily_sentiment_rollup(filtered_sentiment_daily_trend: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_daily_trend.empty or "collection_date" not in filtered_sentiment_daily_trend.columns:
    return pd.DataFrame()

  working = filtered_sentiment_daily_trend.copy()
  working["collection_date"] = pd.to_datetime(working["collection_date"], errors="coerce")
  if "comments_count" not in working.columns:
    working["comments_count"] = 0
  if "videos_covered" not in working.columns:
    working["videos_covered"] = 0

  for col in ["comments_count", "videos_covered", "avg_sentiment_score", "positive_ratio", "negative_ratio"]:
    if col in working.columns:
      working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0)

  def _daily_group(group: pd.DataFrame) -> pd.Series:
    comments_count = float(group["comments_count"].sum()) if "comments_count" in group.columns else float(len(group))
    return pd.Series(
      {
        "comments_count": comments_count,
        "videos_covered": float(group["videos_covered"].sum()) if "videos_covered" in group.columns else 0.0,
        "avg_sentiment_score": _weighted_average(group, "avg_sentiment_score", "comments_count"),
        "positive_ratio": _weighted_average(group, "positive_ratio", "comments_count"),
        "negative_ratio": _weighted_average(group, "negative_ratio", "comments_count"),
      }
    )

  daily_rollup = (
    working.groupby("collection_date", as_index=False)
    .apply(_daily_group, include_groups=False)
    .reset_index()
  )

  if "index" in daily_rollup.columns:
    daily_rollup = daily_rollup.drop(columns=["index"])

  daily_rollup = daily_rollup.sort_values("collection_date")
  daily_rollup["collection_date_label"] = daily_rollup["collection_date"].dt.strftime("%Y-%m-%d")
  return daily_rollup


def _build_comment_engagement_daily(filtered_comments: pd.DataFrame) -> pd.DataFrame:
  if filtered_comments.empty:
    return pd.DataFrame()

  working = filtered_comments.copy()

  if "comment_published_at" not in working.columns:
    return pd.DataFrame()

  working["comment_published_at"] = pd.to_datetime(
    working["comment_published_at"],
    errors="coerce",
    utc=True,
  )
  working = working.dropna(subset=["comment_published_at"]).copy()

  if working.empty:
    return pd.DataFrame()

  like_col = resolve_like_count_column(working)
  if like_col is None:
    working["__resolved_like_count"] = 0
  else:
    working["__resolved_like_count"] = pd.to_numeric(
      working[like_col],
      errors="coerce",
    ).fillna(0)

  working["comment_date"] = working["comment_published_at"].dt.floor("D")

  engagement_daily = (
    working.groupby("comment_date", as_index=False)
    .agg(
      matched_comments=("comment_date", "size"),
      total_comment_likes=("__resolved_like_count", "sum"),
      avg_likes_per_comment=("__resolved_like_count", "mean"),
    )
    .sort_values("comment_date")
  )

  engagement_daily["comment_date_label"] = engagement_daily["comment_date"].dt.strftime("%Y-%m-%d")
  return engagement_daily


def _build_topic_engagement_view(
  filtered_sentiment_topic_summary: pd.DataFrame,
  filtered_comments: pd.DataFrame,
) -> pd.DataFrame:
  topic_summary = filtered_sentiment_topic_summary.copy()
  comments = filtered_comments.copy()

  comment_topic_agg = pd.DataFrame()
  if not comments.empty and "topic" in comments.columns and "genre" in comments.columns:
    like_col = resolve_like_count_column(comments)
    if like_col is None:
      comments["__resolved_like_count"] = 0
    else:
      comments["__resolved_like_count"] = pd.to_numeric(
        comments[like_col],
        errors="coerce",
      ).fillna(0)

    agg_dict: dict[str, tuple[str, str]] = {
      "matched_comments": ("topic", "size"),
      "total_comment_likes": ("__resolved_like_count", "sum"),
      "avg_likes_per_comment": ("__resolved_like_count", "mean"),
    }

    if "video_id" in comments.columns:
      agg_dict["unique_videos"] = ("video_id", "nunique")
    elif "video_title" in comments.columns:
      agg_dict["unique_videos"] = ("video_title", "nunique")

    comment_topic_agg = (
      comments.groupby(["topic", "genre"], as_index=False)
      .agg(**agg_dict)
      .sort_values("matched_comments", ascending=False)
    )

  if topic_summary.empty and comment_topic_agg.empty:
    return pd.DataFrame()

  if not topic_summary.empty:
    for col in [
      "comments_count",
      "avg_sentiment_score",
      "positive_ratio",
      "negative_ratio",
      "total_comment_likes",
      "avg_comment_likes",
      "avg_likes_per_comment",
    ]:
      if col in topic_summary.columns:
        topic_summary[col] = pd.to_numeric(topic_summary[col], errors="coerce").fillna(0)

  if topic_summary.empty:
    merged = comment_topic_agg.copy()
    merged["comments_count"] = merged["matched_comments"]
    merged["avg_sentiment_score"] = 0.0
    merged["positive_ratio"] = 0.0
    merged["negative_ratio"] = 0.0

  elif comment_topic_agg.empty:
    merged = topic_summary.copy()
    merged["matched_comments"] = merged["comments_count"] if "comments_count" in merged.columns else 0

    if "total_comment_likes" not in merged.columns:
      merged["total_comment_likes"] = 0.0

    if "avg_likes_per_comment" not in merged.columns:
      if "avg_comment_likes" in merged.columns:
        merged["avg_likes_per_comment"] = merged["avg_comment_likes"]
      else:
        merged["avg_likes_per_comment"] = 0.0

    if "unique_videos" not in merged.columns:
      merged["unique_videos"] = 0

  else:
    merged = topic_summary.merge(
      comment_topic_agg,
      on=["topic", "genre"],
      how="outer",
      suffixes=("_summary", "_comments"),
    )

    merged["comments_count"] = merged.get("comments_count", pd.Series(dtype=float))
    merged["comments_count"] = merged["comments_count"].fillna(
      merged["matched_comments"] if "matched_comments" in merged.columns else 0
    ).fillna(0)

    if "matched_comments" not in merged.columns:
      merged["matched_comments"] = merged["comments_count"]
    else:
      merged["matched_comments"] = merged["matched_comments"].fillna(merged["comments_count"]).fillna(0)

    if "avg_sentiment_score" not in merged.columns:
      merged["avg_sentiment_score"] = 0.0
    else:
      merged["avg_sentiment_score"] = pd.to_numeric(
        merged["avg_sentiment_score"], errors="coerce"
      ).fillna(0.0)

    if "positive_ratio" not in merged.columns:
      merged["positive_ratio"] = 0.0
    else:
      merged["positive_ratio"] = pd.to_numeric(
        merged["positive_ratio"], errors="coerce"
      ).fillna(0.0)

    if "negative_ratio" not in merged.columns:
      merged["negative_ratio"] = 0.0
    else:
      merged["negative_ratio"] = pd.to_numeric(
        merged["negative_ratio"], errors="coerce"
      ).fillna(0.0)

    like_total_candidates = [
      "total_comment_likes_comments",
      "total_comment_likes_summary",
      "total_comment_likes",
    ]
    merged["total_comment_likes"] = 0.0
    for col in like_total_candidates:
      if col in merged.columns:
        merged["total_comment_likes"] = merged["total_comment_likes"].fillna(0.0) + pd.to_numeric(
          merged[col], errors="coerce"
        ).fillna(0.0)

    avg_like_candidates = [
      "avg_likes_per_comment_comments",
      "avg_likes_per_comment_summary",
      "avg_comment_likes",
      "avg_likes_per_comment",
    ]
    avg_like_source = None
    for col in avg_like_candidates:
      if col in merged.columns:
        avg_like_source = col
        break

    if avg_like_source is None:
      merged["avg_likes_per_comment"] = 0.0
    else:
      merged["avg_likes_per_comment"] = pd.to_numeric(
        merged[avg_like_source], errors="coerce"
      ).fillna(0.0)

    unique_video_candidates = [
      "unique_videos_comments",
      "unique_videos_summary",
      "unique_videos",
    ]
    unique_video_source = None
    for col in unique_video_candidates:
      if col in merged.columns:
        unique_video_source = col
        break

    if unique_video_source is None:
      merged["unique_videos"] = 0
    else:
      merged["unique_videos"] = pd.to_numeric(
        merged[unique_video_source], errors="coerce"
      ).fillna(0)

  merged["topic_display"] = merged["topic"].map(_pretty_text)
  merged["genre_display"] = merged["genre"].map(_pretty_text)
  return merged

def _build_video_evidence_view(
  filtered_comments: pd.DataFrame,
  filtered_sentiment_video_summary: pd.DataFrame,
) -> pd.DataFrame:
  video_summary = filtered_sentiment_video_summary.copy()
  comments = filtered_comments.copy()

  comment_video_agg = pd.DataFrame()
  if not comments.empty:
    group_cols = []
    if "video_id" in comments.columns:
      group_cols.append("video_id")
    if "video_title" in comments.columns:
      group_cols.append("video_title")
    if "channel_title" in comments.columns:
      group_cols.append("channel_title")
    if "topic" in comments.columns:
      group_cols.append("topic")
    if "genre" in comments.columns:
      group_cols.append("genre")

    if group_cols:
      like_col = resolve_like_count_column(comments)
      if like_col is None:
        comments["__resolved_like_count"] = 0
      else:
        comments["__resolved_like_count"] = pd.to_numeric(
          comments[like_col],
          errors="coerce",
        ).fillna(0)

      comment_video_agg = (
        comments.groupby(group_cols, as_index=False)
        .agg(
          matched_comments=("__resolved_like_count", "size"),
          total_comment_likes=("__resolved_like_count", "sum"),
          avg_likes_per_comment=("__resolved_like_count", "mean"),
        )
      )

  if video_summary.empty and comment_video_agg.empty:
    return pd.DataFrame()

  if not video_summary.empty:
    for col in ["comments_count", "avg_sentiment_score", "positive_ratio", "negative_ratio"]:
      if col in video_summary.columns:
        video_summary[col] = pd.to_numeric(video_summary[col], errors="coerce").fillna(0)

  merge_keys = []
  if not video_summary.empty and not comment_video_agg.empty:
    if "video_id" in video_summary.columns and "video_id" in comment_video_agg.columns:
      merge_keys = ["video_id"]
    elif "video_title" in video_summary.columns and "video_title" in comment_video_agg.columns:
      merge_keys = ["video_title"]

  if video_summary.empty:
    merged = comment_video_agg.copy()
    merged["comments_count"] = merged["matched_comments"]
    merged["avg_sentiment_score"] = 0.0
  elif comment_video_agg.empty:
    merged = video_summary.copy()
    merged["matched_comments"] = merged["comments_count"] if "comments_count" in merged.columns else 0
    merged["total_comment_likes"] = 0.0
    merged["avg_likes_per_comment"] = 0.0
  elif merge_keys:
    merged = video_summary.merge(comment_video_agg, on=merge_keys, how="left", suffixes=("", "_comment"))
    if "topic_comment" in merged.columns and "topic" in merged.columns:
      merged["topic"] = merged["topic"].fillna(merged["topic_comment"])
    if "genre_comment" in merged.columns and "genre" in merged.columns:
      merged["genre"] = merged["genre"].fillna(merged["genre_comment"])
    if "channel_title_comment" in merged.columns and "channel_title" in merged.columns:
      merged["channel_title"] = merged["channel_title"].fillna(merged["channel_title_comment"])
    merged["matched_comments"] = merged["matched_comments"].fillna(merged["comments_count"]).fillna(0)
    merged["total_comment_likes"] = merged["total_comment_likes"].fillna(0.0)
    merged["avg_likes_per_comment"] = merged["avg_likes_per_comment"].fillna(0.0)
  else:
    merged = video_summary.copy()
    merged["matched_comments"] = merged["comments_count"] if "comments_count" in merged.columns else 0
    merged["total_comment_likes"] = 0.0
    merged["avg_likes_per_comment"] = 0.0

  if "video_title" not in merged.columns:
    merged["video_title"] = "Unknown Video"
  if "channel_title" not in merged.columns:
    merged["channel_title"] = "Unknown Channel"
  if "topic" not in merged.columns:
    merged["topic"] = "unknown"
  if "genre" not in merged.columns:
    merged["genre"] = "unknown"

  merged["video_title_short"] = merged["video_title"].map(lambda value: _truncate_text(value, 38))
  merged["topic_display"] = merged["topic"].map(_pretty_text)
  merged["genre_display"] = merged["genre"].map(_pretty_text)
  return merged

def _render_snapshot_cards(
  filtered_comments: pd.DataFrame,
  title: str,
  positive: bool,
) -> None:
  if filtered_comments.empty:
    return

  required_columns = {"comment_text", "sentiment_score"}
  if not required_columns.issubset(set(filtered_comments.columns)):
    return

  working = filtered_comments.copy()
  working["sentiment_score"] = pd.to_numeric(working["sentiment_score"], errors="coerce")

  like_col = resolve_like_count_column(working)
  if like_col is None:
    working["__resolved_like_count"] = 0
  else:
    working["__resolved_like_count"] = pd.to_numeric(
      working[like_col],
      errors="coerce",
    ).fillna(0)

  if positive:
    working = working[working["sentiment_score"] > 0]
    working = working.sort_values(["__resolved_like_count", "sentiment_score"], ascending=[False, False])
  else:
    working = working[working["sentiment_score"] < 0]
    working = working.sort_values(["__resolved_like_count", "sentiment_score"], ascending=[False, True])

  if working.empty:
    return

  st.markdown(f"### {title}")
  columns = st.columns(2)
  preview = working.head(4).reset_index(drop=True)

  for idx, (_, row) in enumerate(preview.iterrows()):
    target_col = columns[idx % 2]
    topic_text = _pretty_text(row["topic"]) if "topic" in row.index else "Unknown"
    genre_text = _pretty_text(row["genre"]) if "genre" in row.index else "Unknown"
    video_text = _truncate_text(row["video_title"], 55) if "video_title" in row.index else "Unknown Video"
    like_text = f"{int(row['__resolved_like_count']):,}"

    target_col.markdown(
      f"""
<div style="padding:14px 16px; border:1px solid rgba(255,255,255,0.08); border-radius:14px; margin-bottom:12px; background:rgba(255,255,255,0.02);">
  <div style="font-size:0.92rem; opacity:0.88; margin-bottom:6px;"><b>{topic_text}</b> · {genre_text} · score {row['sentiment_score']:.3f} · likes {like_text}</div>
  <div style="font-size:1rem; margin-bottom:8px;">“{str(row['comment_text'])[:220]}”</div>
  <div style="font-size:0.9rem; opacity:0.78;">{video_text}</div>
</div>
""",
      unsafe_allow_html=True,
    )

def render_descriptive_tab(
  filtered_collection: pd.DataFrame,
  filtered_comments: pd.DataFrame,
  filtered_sentiment_topic_summary: pd.DataFrame,
  filtered_sentiment_daily_trend: pd.DataFrame,
  filtered_sentiment_video_summary: pd.DataFrame,
  filtered_sentiment_weekday_hour: pd.DataFrame,
  filtered_sentiment_keyword: pd.DataFrame,
  weekday_order: list[str],
  format_number: Callable[[float | int], str],
  format_hour_12: Callable[[int], str],
  format_score: Callable[[float | int], str],
  format_pct: Callable[[float | int], str],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Descriptive Analytics")
  st.caption(
    "This page explains what viewers are saying, how audience mood is changing, where engagement is strongest, and which topics or videos are drawing the most attention in the current view."
  )

  has_descriptive_data = not (
    filtered_comments.empty
    and filtered_sentiment_topic_summary.empty
    and filtered_sentiment_daily_trend.empty
    and filtered_sentiment_video_summary.empty
    and filtered_sentiment_weekday_hour.empty
    and filtered_sentiment_keyword.empty
    and filtered_collection.empty
  )

  if not has_descriptive_data:
    show_empty_state("No descriptive data is available for the selected filters.")
    return

  daily_sentiment = _build_daily_sentiment_rollup(filtered_sentiment_daily_trend)
  daily_engagement = _build_comment_engagement_daily(filtered_comments)
  topic_engagement = _build_topic_engagement_view(filtered_sentiment_topic_summary, filtered_comments)
  video_evidence = _build_video_evidence_view(filtered_comments, filtered_sentiment_video_summary)

  matched_comments = int(len(filtered_comments)) if not filtered_comments.empty else (
    int(daily_sentiment["comments_count"].sum()) if not daily_sentiment.empty else 0
  )

  if "video_id" in filtered_comments.columns:
    unique_videos = int(filtered_comments["video_id"].nunique())
  elif "video_id" in filtered_sentiment_video_summary.columns:
    unique_videos = int(filtered_sentiment_video_summary["video_id"].nunique())
  elif not filtered_sentiment_video_summary.empty:
    unique_videos = int(len(filtered_sentiment_video_summary))
  else:
    unique_videos = int(topic_engagement["unique_videos"].sum()) if "unique_videos" in topic_engagement.columns else 0

  avg_sentiment_value = (
    _weighted_average(topic_engagement, "avg_sentiment_score", "matched_comments")
    if not topic_engagement.empty
    else (_weighted_average(daily_sentiment, "avg_sentiment_score", "comments_count") if not daily_sentiment.empty else 0.0)
  )

  positive_ratio_value = (
    _weighted_average(topic_engagement, "positive_ratio", "matched_comments")
    if not topic_engagement.empty
    else (_weighted_average(daily_sentiment, "positive_ratio", "comments_count") if not daily_sentiment.empty else 0.0)
  )

  like_col = resolve_like_count_column(filtered_comments)
  total_comment_likes = (
    float(pd.to_numeric(filtered_comments[like_col], errors="coerce").fillna(0).sum())
    if not filtered_comments.empty and like_col is not None
    else 0.0
  )

  avg_likes_per_comment = _safe_divide(total_comment_likes, float(matched_comments))

  k1, k2, k3, k4, k5, k6 = st.columns(6)
  k1.metric("Matched Comments", format_number(matched_comments))
  k2.metric("Unique Videos", format_number(unique_videos))
  k3.metric("Average Sentiment", format_score(avg_sentiment_value))
  k4.metric("Positive Share", format_pct(positive_ratio_value))
  k5.metric("Total Comment Likes", format_number(int(total_comment_likes)))
  k6.metric("Average Likes per Comment", format_score(avg_likes_per_comment))

  trend_left, trend_right = st.columns(2)

  with trend_left:
    st.markdown("#### Audience Mood Trend Over Time")
    st.caption("This trend shows how the average audience sentiment is moving day by day in the current filtered view.")
    if daily_sentiment.empty:
      st.info("Sentiment trend data is not available for the selected filters.")
    else:
      fig_sentiment_trend = px.line(
        daily_sentiment,
        x="collection_date_label",
        y="avg_sentiment_score",
        markers=True,
        template="plotly_dark",
        title="Audience Mood Trend Over Time",
      )
      fig_sentiment_trend.update_traces(
        hovertemplate=(
          "Date: %{x}<br>"
          "Average sentiment: %{y:.3f}<br>"
          "Matched comments: %{customdata[0]:,.0f}<extra></extra>"
        ),
        customdata=daily_sentiment[["comments_count"]],
      )
      fig_sentiment_trend.update_layout(
        xaxis_title="Collection Date",
        yaxis_title="Average Sentiment Score",
        height=430,
      )
      st.plotly_chart(fig_sentiment_trend, use_container_width=True)

  with trend_right:
    st.markdown("#### Audience Engagement Trend Over Time")
    st.caption("This trend shows how audience interaction is changing over time using total likes on collected comments.")
    if daily_engagement.empty:
      st.info("Engagement trend data is not available for the selected filters.")
    else:
      fig_engagement_trend = px.area(
        daily_engagement,
        x="comment_date_label",
        y="total_comment_likes",
        template="plotly_dark",
        title="Audience Engagement Trend Over Time",
      )
      fig_engagement_trend.update_traces(
        hovertemplate=(
          "Date: %{x}<br>"
          "Total comment likes: %{y:,.0f}<br>"
          "Matched comments: %{customdata[0]:,.0f}<br>"
          "Average likes per comment: %{customdata[1]:.2f}<extra></extra>"
        ),
        customdata=daily_engagement[["matched_comments", "avg_likes_per_comment"]],
      )
      fig_engagement_trend.update_layout(
        xaxis_title="Collection Date",
        yaxis_title="Total Comment Likes",
        height=430,
      )
      st.plotly_chart(fig_engagement_trend, use_container_width=True)

  compare_left, compare_right = st.columns(2)

  with compare_left:
    st.markdown("#### Sentiment versus Engagement by Topic")
    st.caption("Topics in the upper-right area are both more engaging and more positive. Larger bubbles indicate heavier audience discussion.")
    if topic_engagement.empty:
      st.info("Topic-level sentiment and engagement data is not available for the selected filters.")
    else:
      scatter_df = (
        topic_engagement[
          ["topic_display", "genre_display", "avg_sentiment_score", "avg_likes_per_comment", "matched_comments"]
        ]
        .copy()
        .sort_values("matched_comments", ascending=False)
      )

      fig_scatter = px.scatter(
        scatter_df,
        x="avg_sentiment_score",
        y="avg_likes_per_comment",
        size="matched_comments",
        color="genre_display",
        hover_name="topic_display",
        template="plotly_dark",
        title="Sentiment versus Engagement by Topic",
        size_max=42,
      )
      fig_scatter.update_traces(
        hovertemplate=(
          "Topic: %{hovertext}<br>"
          "Genre: %{customdata[0]}<br>"
          "Average sentiment: %{x:.3f}<br>"
          "Average likes per comment: %{y:.2f}<br>"
          "Matched comments: %{marker.size:,.0f}<extra></extra>"
        ),
        customdata=scatter_df[["genre_display"]],
      )
      fig_scatter.update_layout(
        xaxis_title="Average Sentiment Score",
        yaxis_title="Average Likes per Comment",
        legend_title="Genre",
        height=430,
      )
      st.plotly_chart(fig_scatter, use_container_width=True)

  with compare_right:
    st.markdown("#### Audience Attention Map")
    st.caption("This treemap shows where discussion is concentrated across genres and topics in the current view.")
    if topic_engagement.empty:
      st.info("Audience composition data is not available for the selected filters.")
    else:
      treemap_df = topic_engagement.copy()
      treemap_df["size_metric"] = treemap_df["matched_comments"].clip(lower=1)

      fig_treemap = px.treemap(
        treemap_df,
        path=["genre_display", "topic_display"],
        values="size_metric",
        color="avg_sentiment_score",
        color_continuous_scale="RdYlGn",
        template="plotly_dark",
        title="Audience Attention Map",
      )
      fig_treemap.update_traces(
        hovertemplate=(
          "Genre / Topic: %{label}<br>"
          "Matched comments: %{value:,.0f}<br>"
          "Average sentiment: %{color:.3f}<extra></extra>"
        )
      )
      fig_treemap.update_layout(height=430)
      st.plotly_chart(fig_treemap, use_container_width=True)

  st.markdown("#### Comment Engagement Heatmap by Weekday and Hour")
  st.caption("This chart stays exactly focused on when discussion is most active in the selected view, which helps explain timing patterns in audience response.")

  if filtered_comments.empty or "comment_published_at" not in filtered_comments.columns:
    st.info("No comment-level timestamp data is available for the selected filters.")
  else:
    heatmap_comments = filtered_comments.copy()
    heatmap_comments["comment_published_at"] = pd.to_datetime(
      heatmap_comments["comment_published_at"],
      errors="coerce",
      utc=True,
    )

    if "weekday_name" not in heatmap_comments.columns:
      heatmap_comments["weekday_name"] = pd.Categorical(
        heatmap_comments["comment_published_at"].dt.day_name(),
        categories=weekday_order,
        ordered=True,
      )
    if "comment_hour_24" not in heatmap_comments.columns:
      heatmap_comments["comment_hour_24"] = heatmap_comments["comment_published_at"].dt.hour

    hour_heatmap = (
      heatmap_comments
      .dropna(subset=["weekday_name", "comment_hour_24"])
      .groupby(["weekday_name", "comment_hour_24"], as_index=False)
      .size()
      .rename(columns={"size": "comment_volume"})
    )

    if hour_heatmap.empty:
      st.info("No hour-level engagement pattern is available for the selected filters.")
    else:
      pivot_heatmap = hour_heatmap.pivot(
        index="weekday_name",
        columns="comment_hour_24",
        values="comment_volume",
      ).fillna(0)

      valid_weekdays = [day for day in weekday_order if day in pivot_heatmap.index]
      pivot_heatmap = pivot_heatmap.reindex(valid_weekdays)
      pivot_heatmap = pivot_heatmap.reindex(columns=list(range(24)), fill_value=0)
      pivot_heatmap.columns = [format_hour_12(hour_value) for hour_value in pivot_heatmap.columns]  # type: ignore

      fig_hour_heatmap = px.imshow(
        pivot_heatmap,
        aspect="auto",
        template="plotly_dark",
        title="Comment Engagement Heatmap by Weekday and Hour",
      )
      fig_hour_heatmap.update_traces(
        hovertemplate=(
          "Weekday: %{y}<br>"
          "Hour: %{x}<br>"
          "Comments observed: %{z}<extra></extra>"
        )
      )
      fig_hour_heatmap.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Weekday",
        height=520,
      )
      st.plotly_chart(fig_hour_heatmap, use_container_width=True)

  keyword_left, keyword_right = st.columns(2)
  keyword_count_col = resolve_keyword_count_column(filtered_sentiment_keyword)

  if not filtered_sentiment_keyword.empty and keyword_count_col and "keyword" in filtered_sentiment_keyword.columns:
    keyword_working = filtered_sentiment_keyword.copy()
    keyword_working[keyword_count_col] = pd.to_numeric(keyword_working[keyword_count_col], errors="coerce").fillna(0)

    with keyword_left:
      st.markdown("#### Positive Driver Words")
      st.caption("These are the words most frequently associated with positive audience sentiment in the current filtered view.")
      positive_words = keyword_working.copy()
      if "avg_sentiment_score" in positive_words.columns:
        positive_words["avg_sentiment_score"] = pd.to_numeric(
          positive_words["avg_sentiment_score"],
          errors="coerce",
        ).fillna(0)
        positive_words = positive_words[positive_words["avg_sentiment_score"] > 0]

      positive_words = (
        positive_words.sort_values([keyword_count_col, "avg_sentiment_score"], ascending=[False, False])
        .head(12)
      )

      if positive_words.empty:
        st.info("No strong positive driver words are available for the selected filters.")
      else:
        fig_positive_words = px.bar(
          positive_words.sort_values(keyword_count_col, ascending=True),
          x=keyword_count_col,
          y="keyword",
          color="avg_sentiment_score" if "avg_sentiment_score" in positive_words.columns else None,
          orientation="h",
          template="plotly_dark",
          title="Positive Driver Words",
          color_continuous_scale="YlGn",
        )
        fig_positive_words.update_layout(
          xaxis_title="Frequency",
          yaxis_title="Keyword",
          height=420,
          coloraxis_colorbar_title="Avg Sentiment",
        )
        st.plotly_chart(fig_positive_words, use_container_width=True)

    with keyword_right:
      st.markdown("#### Negative Driver Words")
      st.caption("These are the words most frequently associated with negative audience sentiment in the current filtered view.")
      negative_words = keyword_working.copy()
      if "avg_sentiment_score" in negative_words.columns:
        negative_words["avg_sentiment_score"] = pd.to_numeric(
          negative_words["avg_sentiment_score"],
          errors="coerce",
        ).fillna(0)
        negative_words = negative_words[negative_words["avg_sentiment_score"] < 0]

      negative_words = (
        negative_words.sort_values([keyword_count_col, "avg_sentiment_score"], ascending=[False, True])
        .head(12)
      )

      if negative_words.empty:
        st.info("No strong negative driver words are available for the selected filters.")
      else:
        fig_negative_words = px.bar(
          negative_words.sort_values(keyword_count_col, ascending=True),
          x=keyword_count_col,
          y="keyword",
          color="avg_sentiment_score" if "avg_sentiment_score" in negative_words.columns else None,
          orientation="h",
          template="plotly_dark",
          title="Negative Driver Words",
          color_continuous_scale="Reds_r",
        )
        fig_negative_words.update_layout(
          xaxis_title="Frequency",
          yaxis_title="Keyword",
          height=420,
          coloraxis_colorbar_title="Avg Sentiment",
        )
        st.plotly_chart(fig_negative_words, use_container_width=True)
  else:
    keyword_left.info("Keyword-level descriptive data is not available for the selected filters.")
    keyword_right.info("Keyword-level descriptive data is not available for the selected filters.")

  video_left, video_right = st.columns(2)

  if not video_evidence.empty:
    with video_left:
      st.markdown("#### Most Engaging Positive Videos")
      st.caption("These videos combine stronger positive sentiment with stronger audience interaction, so they represent the best-performing content in the current view.")
      positive_videos = (
        video_evidence.sort_values(
          ["avg_sentiment_score", "total_comment_likes", "matched_comments"],
          ascending=[False, False, False],
        )
        .head(10)
        .copy()
      )

      if positive_videos.empty:
        st.info("No positive video evidence is available for the selected filters.")
      else:
        fig_positive_videos = px.bar(
          positive_videos.sort_values("avg_sentiment_score", ascending=True),
          x="avg_sentiment_score",
          y="video_title_short",
          color="genre_display",
          orientation="h",
          hover_data={
            "video_title": True,
            "channel_title": True,
            "matched_comments": True,
            "total_comment_likes": True,
            "video_title_short": False,
            "genre_display": False,
          },
          template="plotly_dark",
          title="Most Engaging Positive Videos",
        )
        fig_positive_videos.update_layout(
          xaxis_title="Average Sentiment Score",
          yaxis_title="Video",
          legend_title="Genre",
          height=480,
        )
        st.plotly_chart(fig_positive_videos, use_container_width=True)

    with video_right:
      st.markdown("#### Most Engaging Risk Videos")
      st.caption("These videos are drawing attention but also showing weaker sentiment, so they are the strongest candidates for closer review.")
      risk_videos = (
        video_evidence.sort_values(
          ["avg_sentiment_score", "total_comment_likes", "matched_comments"],
          ascending=[True, False, False],
        )
        .head(10)
        .copy()
      )

      if risk_videos.empty:
        st.info("No risk video evidence is available for the selected filters.")
      else:
        fig_risk_videos = px.bar(
          risk_videos.sort_values("avg_sentiment_score", ascending=False),
          x="avg_sentiment_score",
          y="video_title_short",
          color="genre_display",
          orientation="h",
          hover_data={
            "video_title": True,
            "channel_title": True,
            "matched_comments": True,
            "total_comment_likes": True,
            "video_title_short": False,
            "genre_display": False,
          },
          template="plotly_dark",
          title="Most Engaging Risk Videos",
        )
        fig_risk_videos.update_layout(
          xaxis_title="Average Sentiment Score",
          yaxis_title="Video",
          legend_title="Genre",
          height=480,
        )
        st.plotly_chart(fig_risk_videos, use_container_width=True)

  _render_snapshot_cards(
    filtered_comments=filtered_comments,
    title="Audience-Endorsed Positive Comments",
    positive=True,
  )
  _render_snapshot_cards(
    filtered_comments=filtered_comments,
    title="Audience-Endorsed Negative Comments",
    positive=False,
  )