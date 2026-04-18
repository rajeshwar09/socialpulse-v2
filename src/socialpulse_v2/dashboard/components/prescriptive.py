from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st


def render_prescriptive_tab(
  filtered_query: pd.DataFrame,
  build_prescriptive_recommendations: Callable[[pd.DataFrame], list[str]],
) -> None:
  st.subheader("Prescriptive Analytics")

  recommendations = build_prescriptive_recommendations(filtered_query)

  for index, recommendation in enumerate(recommendations, start=1):
    st.markdown(f"**Recommendation {index}.** {recommendation}")