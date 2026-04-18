from __future__ import annotations

from typing import Callable

import pandas as pd
import plotly.express as px
import streamlit as st


def render_predictive_tab(
  filtered_collection: pd.DataFrame,
  show_empty_state: Callable[[str], None],
) -> None:
  st.subheader("Predictive Analytics")

  if filtered_collection.empty:
    show_empty_state("Not enough filtered data is available for predictive analytics.")
    return

  predictive_df = (
    filtered_collection
    .groupby("run_date_label", as_index=False)["total_records_written"]
    .sum()
    .sort_values("run_date_label")
  ) # type: ignore

  predictive_df["forecast_baseline"] = predictive_df["total_records_written"].rolling(
    window=min(3, len(predictive_df)),
    min_periods=1,
  ).mean()

  fig_predictive = px.line(
    predictive_df,
    x="run_date_label",
    y=["total_records_written", "forecast_baseline"],
    template="plotly_dark",
    title="Observed Volume versus Rolling Forecast Baseline",
  )
  fig_predictive.update_layout(
    xaxis_title="Run Date",
    yaxis_title="Written Records",
    legend_title="Series",
    height=450,
  )
  st.plotly_chart(fig_predictive, use_container_width=True)

  st.markdown(
    "This predictive view currently shows a rolling baseline forecast. As historical depth grows, it can be upgraded to a stronger forecasting model."
  )