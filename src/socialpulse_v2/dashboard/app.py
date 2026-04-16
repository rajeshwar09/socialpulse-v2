import streamlit as st

from socialpulse_v2.core.settings import settings


st.set_page_config(
  page_title=settings.app_name,
  page_icon="📊",
  layout="wide",
)

st.title("SocialPulse V2")
st.subheader("Query-driven YouTube analytics platform")

st.markdown(
  """
  ### Phase 00 status
  The project scaffold is ready.

  Upcoming implementation:
  - Lakehouse foundation
  - Historical JSON bootstrap
  - Query-driven daily ingestion
  - ML sentiment engine
  - Descriptive / Diagnostic / Predictive / Prescriptive dashboard
  """
)