from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
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


def _build_forecast(history_df: pd.DataFrame, periods: int = 7) -> pd.DataFrame:
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
    }
  )

  return forecast_df


def render_predictive_tab(
  filtered_sentiment_daily_trend: pd.DataFrame,
  format_number: Callable[[float | int], str],
  format_score: Callable[[float | int], str],
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Predictive Analytics")
  st.caption(
    "This outlook uses a damped trend forecast from daily comment sentiment history, so it is smoother than a plain straight-line projection."
  )

  history_df = _build_daily_history(filtered_sentiment_daily_trend)

  if history_df.empty:
    show_empty_state("Not enough filtered sentiment history is available for predictive analytics.")
    return

  forecast_df = _build_forecast(history_df, periods=7)

  latest_comments = int(history_df.iloc[-1]["comments_count"])
  latest_sentiment = float(history_df.iloc[-1]["avg_sentiment_score"])
  forecast_comments_mean = float(forecast_df["forecast_comment_count"].mean()) if not forecast_df.empty else 0.0
  forecast_sentiment_mean = float(forecast_df["forecast_sentiment_score"].mean()) if not forecast_df.empty else 0.0

  c1, c2, c3, c4 = st.columns(4)
  c1.metric("Latest Daily Comments", format_number(latest_comments))
  c2.metric("7-Day Forecast Mean", format_number(round(forecast_comments_mean)))
  c3.metric("Latest Sentiment", format_score(latest_sentiment))
  c4.metric("Forecast Sentiment Mean", format_score(forecast_sentiment_mean))

  comment_fig = go.Figure()
  comment_fig.add_trace(
    go.Scatter(
      x=history_df["collection_date"],
      y=history_df["comments_count"],
      mode="lines+markers",
      name="Actual",
    )
  )

  if not forecast_df.empty:
    comment_fig.add_trace(
      go.Scatter(
        x=forecast_df["forecast_date"],
        y=forecast_df["comments_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
      )
    )
    comment_fig.add_trace(
      go.Scatter(
        x=forecast_df["forecast_date"],
        y=forecast_df["comments_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Forecast Range",
      )
    )
    comment_fig.add_trace(
      go.Scatter(
        x=forecast_df["forecast_date"],
        y=forecast_df["forecast_comment_count"],
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

  sentiment_fig = go.Figure()
  sentiment_fig.add_trace(
    go.Scatter(
      x=history_df["collection_date"],
      y=history_df["avg_sentiment_score"],
      mode="lines+markers",
      name="Actual",
    )
  )

  if not forecast_df.empty:
    sentiment_fig.add_trace(
      go.Scatter(
        x=forecast_df["forecast_date"],
        y=forecast_df["sentiment_upper"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
      )
    )
    sentiment_fig.add_trace(
      go.Scatter(
        x=forecast_df["forecast_date"],
        y=forecast_df["sentiment_lower"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        name="Forecast Range",
      )
    )
    sentiment_fig.add_trace(
      go.Scatter(
        x=forecast_df["forecast_date"],
        y=forecast_df["forecast_sentiment_score"],
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

  if forecast_sentiment_mean > latest_sentiment + 0.03:
    st.success("Forecast suggests sentiment may improve over the next few days.")
  elif forecast_sentiment_mean < latest_sentiment - 0.03:
    st.warning("Forecast suggests sentiment may weaken unless content response improves.")
  else:
    st.info("Forecast suggests sentiment may remain relatively stable in the near term.")