from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def _build_daily_history(filtered_sentiment_daily_trend: pd.DataFrame) -> pd.DataFrame:
  if filtered_sentiment_daily_trend.empty:
    return pd.DataFrame(columns=["collection_date", "comments_count", "avg_sentiment_score"])

  df = filtered_sentiment_daily_trend.copy()
  df["collection_date"] = pd.to_datetime(df["collection_date"], errors="coerce")
  df = df.dropna(subset=["collection_date"])

  if df.empty:
    return pd.DataFrame(columns=["collection_date", "comments_count", "avg_sentiment_score"])

  if "comments_count" not in df.columns:
    df["comments_count"] = 1
  if "avg_sentiment_score" not in df.columns:
    df["avg_sentiment_score"] = 0.0

  out = (
    df.groupby("collection_date", as_index=False)
    .apply(
      lambda group: pd.Series(
        {
          "comments_count": float(group["comments_count"].sum()),
          "avg_sentiment_score": float(
            (group["avg_sentiment_score"] * group["comments_count"].clip(lower=1)).sum()
            / group["comments_count"].clip(lower=1).sum()
          ),
        }
      )
    )
    .reset_index(drop=True)
    .sort_values("collection_date")
  )

  return out


def _damped_trend_forecast(
  values: np.ndarray,
  periods: int,
  alpha: float = 0.60,
  beta: float = 0.35,
  phi: float = 0.75,
  clip_min: float | None = None,
  clip_max: float | None = None,
) -> np.ndarray:
  if len(values) == 0:
    return np.array([])

  if len(values) == 1:
    out = np.repeat(float(values[0]), periods)
  else:
    level = float(values[0])
    trend = float(values[1] - values[0])

    for observed in values[1:]:
      previous_level = level
      level = alpha * float(observed) + (1 - alpha) * (level + phi * trend)
      trend = beta * (level - previous_level) + (1 - beta) * phi * trend

    forecasts: list[float] = []
    current_level = level
    current_trend = trend

    for _ in range(periods):
      current_level = current_level + phi * current_trend
      current_trend = phi * current_trend
      forecasts.append(float(current_level))

    out = np.array(forecasts)

  if clip_min is not None:
    out = np.maximum(out, clip_min)
  if clip_max is not None:
    out = np.minimum(out, clip_max)

  return out


def _build_local_fallback_forecast(history_df: pd.DataFrame, periods: int = 7) -> pd.DataFrame:
  if history_df.empty:
    return pd.DataFrame()

  history = history_df.sort_values("collection_date").copy()

  comment_forecast = _damped_trend_forecast(
    history["comments_count"].astype(float).to_numpy(),
    periods=periods,
    clip_min=0.0,
  )
  sentiment_forecast = _damped_trend_forecast(
    history["avg_sentiment_score"].astype(float).to_numpy(),
    periods=periods,
    clip_min=-1.0,
    clip_max=1.0,
  )

  last_date = history["collection_date"].max()
  future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods + 1)]

  comment_std = max(float(history["comments_count"].std(ddof=0) or 0.0), 1.0)
  sentiment_std = max(float(history["avg_sentiment_score"].std(ddof=0) or 0.0), 0.02)

  forecast_df = pd.DataFrame(
    {
      "forecast_date": future_dates,
      "forecast_comment_count": comment_forecast,
      "forecast_sentiment_score": sentiment_forecast,
      "comments_lower": np.maximum(comment_forecast - comment_std, 0.0),
      "comments_upper": comment_forecast + comment_std,
      "sentiment_lower": np.maximum(sentiment_forecast - sentiment_std, -1.0),
      "sentiment_upper": np.minimum(sentiment_forecast + sentiment_std, 1.0),
      "model_name": "local_damped_trend",
    }
  )

  return forecast_df


def _filter_predictive_tables(
  forecast_summary_df: pd.DataFrame,
  forecast_7d_df: pd.DataFrame,
  filtered_sentiment_comments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  summary = forecast_summary_df.copy()
  forecast = forecast_7d_df.copy()

  if filtered_sentiment_comments.empty:
    return (
      pd.DataFrame(columns=summary.columns),
      pd.DataFrame(columns=forecast.columns),
    )

  allowed_topics = (
    filtered_sentiment_comments["topic"].dropna().astype(str).unique().tolist()
    if "topic" in filtered_sentiment_comments.columns
    else []
  )
  allowed_genres = (
    filtered_sentiment_comments["genre"].dropna().astype(str).unique().tolist()
    if "genre" in filtered_sentiment_comments.columns
    else []
  )

  if not summary.empty:
    if allowed_topics and "topic" in summary.columns:
      summary = summary[summary["topic"].astype(str).isin(allowed_topics)]
    if allowed_genres and "genre" in summary.columns:
      summary = summary[summary["genre"].astype(str).isin(allowed_genres)]

  if not forecast.empty:
    if allowed_topics and "topic" in forecast.columns:
      forecast = forecast[forecast["topic"].astype(str).isin(allowed_topics)]
    if allowed_genres and "genre" in forecast.columns:
      forecast = forecast[forecast["genre"].astype(str).isin(allowed_genres)]

  return summary, forecast


def _aggregate_forecast(forecast_df: pd.DataFrame) -> pd.DataFrame:
  if forecast_df.empty:
    return pd.DataFrame(columns=["forecast_date", "forecast_comment_count", "forecast_sentiment_score"])

  work = forecast_df.copy()
  work["forecast_date"] = pd.to_datetime(work["forecast_date"], errors="coerce")
  work = work.dropna(subset=["forecast_date"])

  if work.empty:
    return pd.DataFrame(columns=["forecast_date", "forecast_comment_count", "forecast_sentiment_score"])

  out = (
    work.groupby("forecast_date", as_index=False)
    .apply(
      lambda group: pd.Series(
        {
          "forecast_comment_count": float(group["forecast_comment_count"].sum()),
          "forecast_sentiment_score": float(
            (group["forecast_sentiment_score"] * group["forecast_comment_count"].clip(lower=1)).sum()
            / group["forecast_comment_count"].clip(lower=1).sum()
          ),
        }
      )
    )
    .reset_index(drop=True)
    .sort_values("forecast_date")
  )

  comment_std = max(float(out["forecast_comment_count"].std(ddof=0) or 0.0), 1.0)
  sentiment_std = max(float(out["forecast_sentiment_score"].std(ddof=0) or 0.0), 0.02)

  out["comments_lower"] = np.maximum(out["forecast_comment_count"] - comment_std, 0.0)
  out["comments_upper"] = out["forecast_comment_count"] + comment_std
  out["sentiment_lower"] = np.maximum(out["forecast_sentiment_score"] - sentiment_std, -1.0)
  out["sentiment_upper"] = np.minimum(out["forecast_sentiment_score"] + sentiment_std, 1.0)

  return out


def _build_genre_forecast_summary(
  filtered_sentiment_comments: pd.DataFrame,
  filtered_forecast_df: pd.DataFrame,
) -> pd.DataFrame:
  columns = [
    "genre",
    "current_comments",
    "current_sentiment",
    "forecast_comments",
    "forecast_sentiment",
    "comments_change_pct",
    "sentiment_change",
  ]

  if filtered_sentiment_comments.empty:
    return pd.DataFrame(columns=columns)

  current_df = filtered_sentiment_comments.copy()
  if "comment_id" not in current_df.columns:
    current_df["comment_id"] = range(len(current_df))
  if "genre" not in current_df.columns:
    current_df["genre"] = "unknown"
  if "sentiment_score" not in current_df.columns:
    current_df["sentiment_score"] = 0.0

  current_summary = (
    current_df.groupby("genre", as_index=False)
    .agg(
      current_comments=("comment_id", "nunique"),
      current_sentiment=("sentiment_score", "mean"),
    )
  )

  if filtered_forecast_df.empty:
    out = current_summary.copy()
    out["forecast_comments"] = np.nan
    out["forecast_sentiment"] = np.nan
    out["comments_change_pct"] = np.nan
    out["sentiment_change"] = np.nan
    return out[columns]

  forecast_df = filtered_forecast_df.copy()
  if "genre" not in forecast_df.columns:
    forecast_df["genre"] = "unknown"

  forecast_summary = (
    forecast_df.groupby("genre", as_index=False)
    .agg(
      forecast_comments=("forecast_comment_count", "mean"),
      forecast_sentiment=("forecast_sentiment_score", "mean"),
    )
  )

  out = current_summary.merge(forecast_summary, on="genre", how="left")

  out["current_comments"] = pd.to_numeric(out["current_comments"], errors="coerce")
  out["current_sentiment"] = pd.to_numeric(out["current_sentiment"], errors="coerce")
  out["forecast_comments"] = pd.to_numeric(out["forecast_comments"], errors="coerce")
  out["forecast_sentiment"] = pd.to_numeric(out["forecast_sentiment"], errors="coerce")

  out["comments_change_pct"] = np.where(
    out["current_comments"].notna() & out["forecast_comments"].notna() & (out["current_comments"] > 0),
    ((out["forecast_comments"] - out["current_comments"]) / out["current_comments"]) * 100.0,
    np.nan,
  )
  out["sentiment_change"] = np.where(
    out["current_sentiment"].notna() & out["forecast_sentiment"].notna(),
    out["forecast_sentiment"] - out["current_sentiment"],
    np.nan,
  )

  return out[columns].sort_values(["forecast_comments", "current_comments"], ascending=[False, False])


def _direction_text(value: float, threshold: float) -> str:
  if pd.isna(value):
    return "not enough forecast evidence"
  if value > threshold:
    return "rising"
  if value < -threshold:
    return "falling"
  return "stable"


def _genre_display(value: object) -> str:
  return str(value).replace("_", " ").title()


def _format_optional_number(value: float | int | None, formatter: Callable[[float | int], str]) -> str:
  if value is None or pd.isna(value):
    return "Not enough forecast evidence"
  return formatter(value)


def _format_optional_score(value: float | int | None, formatter: Callable[[float | int], str]) -> str:
  if value is None or pd.isna(value):
    return "Not enough forecast evidence"
  return formatter(value)


def render_predictive_tab(
  filtered_sentiment_daily_trend: pd.DataFrame,
  filtered_sentiment_comments: pd.DataFrame,
  predictive_forecast_summary_df: pd.DataFrame,
  predictive_forecast_7d_df: pd.DataFrame,
  format_number: Callable[[float | int], str],
  format_score: Callable[[float | int], str],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Predictive Analytics")
  st.caption(
    "If no keyword is typed, this forecast covers the whole current filtered dashboard view. If a keyword is typed, the forecast is limited to that matched keyword/topic/genre scope."
  )

  history_df = _build_daily_history(filtered_sentiment_daily_trend)

  filtered_summary_df, filtered_forecast_df = _filter_predictive_tables(
    forecast_summary_df=predictive_forecast_summary_df,
    forecast_7d_df=predictive_forecast_7d_df,
    filtered_sentiment_comments=filtered_sentiment_comments,
  )

  aggregated_forecast_df = _aggregate_forecast(filtered_forecast_df)
  using_gold_forecast = not aggregated_forecast_df.empty

  if not using_gold_forecast:
    aggregated_forecast_df = _build_local_fallback_forecast(history_df, periods=7)

  if history_df.empty and aggregated_forecast_df.empty:
    show_empty_state("Not enough filtered sentiment history is available for predictive analytics.")
    return

  latest_comments = int(history_df.iloc[-1]["comments_count"]) if not history_df.empty else 0
  latest_sentiment = float(history_df.iloc[-1]["avg_sentiment_score"]) if not history_df.empty else 0.0

  forecast_comments_mean = (
    float(aggregated_forecast_df["forecast_comment_count"].mean())
    if not aggregated_forecast_df.empty else np.nan
  )
  forecast_sentiment_mean = (
    float(aggregated_forecast_df["forecast_sentiment_score"].mean())
    if not aggregated_forecast_df.empty else np.nan
  )

  forecast_ready_topics = 0
  model_name = "local_damped_trend"

  if not filtered_summary_df.empty and "is_forecast_eligible" in filtered_summary_df.columns:
    forecast_ready_topics = int(filtered_summary_df["is_forecast_eligible"].fillna(False).sum())

  if not filtered_forecast_df.empty and "model_name" in filtered_forecast_df.columns:
    non_null_models = filtered_forecast_df["model_name"].dropna()
    if not non_null_models.empty:
      model_name = str(non_null_models.iloc[0])
  elif not filtered_summary_df.empty and "forecast_method" in filtered_summary_df.columns:
    non_null_methods = filtered_summary_df["forecast_method"].dropna()
    if not non_null_methods.empty:
      model_name = str(non_null_methods.iloc[0])

  c1, c2, c3, c4, c5 = st.columns(5)
  c1.metric("Latest Daily Comments", format_number(latest_comments))
  c2.metric(
    "7-Day Forecast Mean",
    _format_optional_number(round(forecast_comments_mean) if not pd.isna(forecast_comments_mean) else np.nan, format_number),
  )
  c3.metric("Latest Sentiment", format_score(latest_sentiment))
  c4.metric(
    "Forecast Sentiment Mean",
    _format_optional_score(forecast_sentiment_mean, format_score),
  )
  c5.metric("Forecast-Ready Topics", format_number(forecast_ready_topics))

  st.caption(f"Predictive model in use: {model_name}")

  volume_delta = (
    forecast_comments_mean - latest_comments
    if not pd.isna(forecast_comments_mean) else np.nan
  )
  sentiment_delta = (
    forecast_sentiment_mean - latest_sentiment
    if not pd.isna(forecast_sentiment_mean) else np.nan
  )

  overall_volume_direction = _direction_text(volume_delta, threshold=5.0)
  overall_sentiment_direction = _direction_text(sentiment_delta, threshold=0.03)

  st.markdown("### Overall Forecast Interpretation")
  st.markdown(
    f"""
<div style="padding:12px 14px; border:1px solid rgba(255,255,255,0.08); border-radius:12px; margin-bottom:10px; background:rgba(255,255,255,0.02);">
  <b>Overall attention outlook:</b> {overall_volume_direction.title()}<br/>
  <b>Overall sentiment outlook:</b> {overall_sentiment_direction.title()}<br/>
  <b>Interpretation:</b> The current filtered view suggests that audience attention is <b>{overall_volume_direction}</b> and sentiment is <b>{overall_sentiment_direction}</b> over the next 7 days.
</div>
""",
    unsafe_allow_html=True,
  )

  comment_fig = go.Figure()

  if not history_df.empty:
    comment_fig.add_trace(
      go.Scatter(
        x=history_df["collection_date"],
        y=history_df["comments_count"],
        mode="lines+markers",
        name="Actual",
      )
    )

  if not aggregated_forecast_df.empty:
    comment_fig.add_trace(
      go.Scatter(
        x=aggregated_forecast_df["forecast_date"],
        y=aggregated_forecast_df["comments_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
      )
    )
    comment_fig.add_trace(
      go.Scatter(
        x=aggregated_forecast_df["forecast_date"],
        y=aggregated_forecast_df["comments_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Forecast Range",
      )
    )
    comment_fig.add_trace(
      go.Scatter(
        x=aggregated_forecast_df["forecast_date"],
        y=aggregated_forecast_df["forecast_comment_count"],
        mode="lines+markers",
        name="Forecast",
      )
    )

  comment_fig.update_layout(
    template="plotly_dark",
    title="Actual versus Forecast Comment Volume",
    xaxis_title="Date",
    yaxis_title="Comments",
    height=430,
    legend_title="Series",
  )
  st.plotly_chart(comment_fig, use_container_width=True)

  st.markdown("### Comment Volume Interpretation")
  st.markdown(
    f"""
<div style="padding:12px 14px; border:1px solid rgba(255,255,255,0.08); border-radius:12px; margin-bottom:10px; background:rgba(255,255,255,0.02);">
  <b>Latest daily comments:</b> {format_number(latest_comments)}<br/>
  <b>Forecast average daily comments:</b> {_format_optional_number(round(forecast_comments_mean) if not pd.isna(forecast_comments_mean) else np.nan, format_number)}<br/>
  <b>Interpretation:</b> Audience attention is expected to remain <b>{overall_volume_direction}</b> in the near-term forecast window.
</div>
""",
    unsafe_allow_html=True,
  )

  sentiment_fig = go.Figure()

  if not history_df.empty:
    sentiment_fig.add_trace(
      go.Scatter(
        x=history_df["collection_date"],
        y=history_df["avg_sentiment_score"],
        mode="lines+markers",
        name="Actual",
      )
    )

  if not aggregated_forecast_df.empty:
    sentiment_fig.add_trace(
      go.Scatter(
        x=aggregated_forecast_df["forecast_date"],
        y=aggregated_forecast_df["sentiment_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
      )
    )
    sentiment_fig.add_trace(
      go.Scatter(
        x=aggregated_forecast_df["forecast_date"],
        y=aggregated_forecast_df["sentiment_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Forecast Range",
      )
    )
    sentiment_fig.add_trace(
      go.Scatter(
        x=aggregated_forecast_df["forecast_date"],
        y=aggregated_forecast_df["forecast_sentiment_score"],
        mode="lines+markers",
        name="Forecast",
      )
    )

  sentiment_fig.update_layout(
    template="plotly_dark",
    title="Actual versus Forecast Sentiment Score",
    xaxis_title="Date",
    yaxis_title="Average Sentiment Score",
    height=430,
    legend_title="Series",
  )
  st.plotly_chart(sentiment_fig, use_container_width=True)

  st.markdown("### Sentiment Interpretation")
  st.markdown(
    f"""
<div style="padding:12px 14px; border:1px solid rgba(255,255,255,0.08); border-radius:12px; margin-bottom:10px; background:rgba(255,255,255,0.02);">
  <b>Latest sentiment score:</b> {format_score(latest_sentiment)}<br/>
  <b>Forecast sentiment score:</b> {_format_optional_score(forecast_sentiment_mean, format_score)}<br/>
  <b>Interpretation:</b> Sentiment is expected to remain <b>{overall_sentiment_direction}</b> in the near-term forecast window.
</div>
""",
    unsafe_allow_html=True,
  )

  if not filtered_forecast_df.empty:
    forecast_by_topic = (
      filtered_forecast_df
      .groupby(["topic", "genre"], as_index=False)
      .agg(
        avg_forecast_comments=("forecast_comment_count", "mean"),
        avg_forecast_sentiment=("forecast_sentiment_score", "mean"),
      )
    )

    forecast_by_topic["topic_display"] = forecast_by_topic["topic"].astype(str).str.replace("_", " ").str.title()
    forecast_by_topic["genre_display"] = forecast_by_topic["genre"].astype(str).str.replace("_", " ").str.title()

    left_col, right_col = st.columns(2)

    rising_topics = forecast_by_topic.sort_values(
      ["avg_forecast_comments", "avg_forecast_sentiment"],
      ascending=[False, False],
    ).head(10)

    fig_rising = px.bar(
      rising_topics,
      x="avg_forecast_comments",
      y="topic_display",
      color="genre_display",
      orientation="h",
      template="plotly_dark",
      title="Topics Expected to Draw More Attention",
    )
    fig_rising.update_layout(
      xaxis_title="Average Forecast Daily Comments",
      yaxis_title="Topic",
      legend_title="Genre",
      height=460,
      yaxis={"categoryorder": "total ascending"},
    )
    left_col.plotly_chart(fig_rising, use_container_width=True)

    risk_topics = forecast_by_topic.sort_values(
      ["avg_forecast_sentiment", "avg_forecast_comments"],
      ascending=[True, False],
    ).head(10)

    fig_risk = px.bar(
      risk_topics,
      x="avg_forecast_sentiment",
      y="topic_display",
      color="genre_display",
      orientation="h",
      template="plotly_dark",
      title="Topics with Forecast Sentiment Risk",
    )
    fig_risk.update_layout(
      xaxis_title="Average Forecast Sentiment",
      yaxis_title="Topic",
      legend_title="Genre",
      height=460,
      yaxis={"categoryorder": "total ascending"},
    )
    right_col.plotly_chart(fig_risk, use_container_width=True)

  genre_summary_df = _build_genre_forecast_summary(
    filtered_sentiment_comments=filtered_sentiment_comments,
    filtered_forecast_df=filtered_forecast_df,
  )

  genre_forecast_available = not genre_summary_df.empty and genre_summary_df["forecast_sentiment"].notna().any()

  if genre_forecast_available:
    st.markdown("### Genre-Wise Forecast Interpretation")

    for _, row in genre_summary_df.sort_values(["forecast_comments", "current_comments"], ascending=[False, False]).head(8).iterrows():
      genre_name = _genre_display(row["genre"])

      volume_direction = _direction_text(row["comments_change_pct"], threshold=10.0)
      sentiment_direction = _direction_text(row["sentiment_change"], threshold=0.03)

      current_comments_text = _format_optional_number(row["current_comments"], format_number)
      forecast_comments_text = _format_optional_number(
        round(row["forecast_comments"]) if not pd.isna(row["forecast_comments"]) else np.nan,
        format_number,
      )
      current_sentiment_text = _format_optional_score(row["current_sentiment"], format_score)
      forecast_sentiment_text = _format_optional_score(row["forecast_sentiment"], format_score)

      st.markdown(
        f"""
<div style="padding:12px 14px; border:1px solid rgba(255,255,255,0.08); border-radius:12px; margin-bottom:10px; background:rgba(255,255,255,0.02);">
  <b>{genre_name}</b><br/>
  Current average daily comments: <b>{current_comments_text}</b> → Forecast average daily comments: <b>{forecast_comments_text}</b><br/>
  Current sentiment: <b>{current_sentiment_text}</b> → Forecast sentiment: <b>{forecast_sentiment_text}</b><br/>
  <b>Interpretation:</b> Attention is <b>{volume_direction}</b> and sentiment is <b>{sentiment_direction}</b> for this genre in the near-term outlook.
</div>
""",
        unsafe_allow_html=True,
      )
  else:
    st.info("Genre-level forecast interpretation is not available for this current filter yet. The charts above are showing the overall filtered-view forecast.")

  if not pd.isna(forecast_sentiment_mean):
    if forecast_sentiment_mean > latest_sentiment + 0.03:
      st.success("Forecast suggests sentiment may improve over the next few days.")
    elif forecast_sentiment_mean < latest_sentiment - 0.03:
      st.warning("Forecast suggests sentiment may weaken unless content response improves.")
    else:
      st.info("Forecast suggests sentiment may remain relatively stable in the near term.")
  else:
    st.info("Not enough forecast evidence is available yet for a reliable sentiment outlook.")