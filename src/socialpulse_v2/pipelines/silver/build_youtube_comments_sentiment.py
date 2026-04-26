from __future__ import annotations

from datetime import UTC, datetime
import os

import pandas as pd

from socialpulse_v2.ml.sentiment.rule_based import score_text as score_text_rule_based
from socialpulse_v2.ml.sentiment.transformer_inference import score_texts as score_texts_transformer


OUTPUT_COLUMNS = [
  "platform",
  "collection_date",
  "query_id",
  "query_text",
  "topic",
  "genre",
  "video_id",
  "video_title",
  "channel_title",
  "video_published_at",
  "comment_id",
  "comment_text",
  "comment_author_name",
  "comment_like_count",
  "reply_count",
  "comment_published_at",
  "language_target",
  "sentiment_backend",
  "sentiment_model",
  "sentiment_score",
  "sentiment_label",
  "sentiment_confidence",
  "sentiment_negative_prob",
  "sentiment_neutral_prob",
  "sentiment_positive_prob",
  "positive_hits",
  "negative_hits",
  "token_count",
  "processed_at",
]


def _now_iso() -> str:
  return datetime.now(UTC).isoformat()


def _resolve_backend() -> str:
  backend = os.getenv("SOCIALPULSE_SENTIMENT_BACKEND", "transformer").strip().lower()
  return backend if backend in {"transformer", "rule_based"} else "transformer"


def _allow_fallback() -> bool:
  value = os.getenv("SOCIALPULSE_SENTIMENT_ALLOW_FALLBACK", "false").strip().lower()
  return value in {"1", "true", "yes", "y"}


def _to_probability_columns(label: str, confidence: float) -> tuple[float, float, float]:
  bounded_confidence = min(max(float(confidence), 0.0), 1.0)

  if label == "positive":
    return (round(1.0 - bounded_confidence, 6), round(0.0, 6), round(bounded_confidence, 6))

  if label == "negative":
    return (round(bounded_confidence, 6), round(0.0, 6), round(1.0 - bounded_confidence, 6))

  return (
    round((1.0 - bounded_confidence) / 2.0, 6),
    round(bounded_confidence, 6),
    round((1.0 - bounded_confidence) / 2.0, 6),
  )


def _score_with_transformer(texts: list[str]) -> list[dict]:
  results = score_texts_transformer(texts)
  rows: list[dict] = []

  for result in results:
    rows.append(
      {
        "sentiment_backend": "transformer",
        "sentiment_model": result.model_name,
        "sentiment_score": result.sentiment_score,
        "sentiment_label": result.sentiment_label,
        "sentiment_confidence": result.sentiment_confidence,
        "sentiment_negative_prob": result.sentiment_negative_prob,
        "sentiment_neutral_prob": result.sentiment_neutral_prob,
        "sentiment_positive_prob": result.sentiment_positive_prob,
        "positive_hits": int(round(result.sentiment_positive_prob * result.token_count)),
        "negative_hits": int(round(result.sentiment_negative_prob * result.token_count)),
        "token_count": int(result.token_count),
      }
    )

  return rows


def _score_with_rule_based(texts: list[str]) -> list[dict]:
  rows: list[dict] = []

  for text in texts:
    result = score_text_rule_based(text)
    negative_prob, neutral_prob, positive_prob = _to_probability_columns(
      label=str(result.label),
      confidence=abs(float(result.score)),
    )

    rows.append(
      {
        "sentiment_backend": "rule_based",
        "sentiment_model": "rule_based_lexicon_v1",
        "sentiment_score": float(result.score),
        "sentiment_label": str(result.label),
        "sentiment_confidence": max(negative_prob, neutral_prob, positive_prob),
        "sentiment_negative_prob": negative_prob,
        "sentiment_neutral_prob": neutral_prob,
        "sentiment_positive_prob": positive_prob,
        "positive_hits": int(result.positive_hits),
        "negative_hits": int(result.negative_hits),
        "token_count": int(result.token_count),
      }
    )

  return rows


def _build_sentiment_rows(texts: list[str]) -> list[dict]:
  backend = _resolve_backend()

  if backend == "rule_based":
    return _score_with_rule_based(texts)

  try:
    return _score_with_transformer(texts)
  except Exception as exc:
    if _allow_fallback():
      print(f"Transformer sentiment inference failed. Falling back to rule-based scoring. Reason: {exc}")
      return _score_with_rule_based(texts)
    raise


def build_youtube_comments_sentiment(bronze_comments_df: pd.DataFrame) -> pd.DataFrame:
  if bronze_comments_df.empty:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)

  working_df = bronze_comments_df.copy()

  for column in OUTPUT_COLUMNS:
    if column not in working_df.columns:
      working_df[column] = pd.NA

  working_df["comment_text"] = working_df["comment_text"].fillna("").astype(str)
  working_df["processed_at"] = _now_iso()

  sentiment_rows = _build_sentiment_rows(working_df["comment_text"].tolist())
  sentiment_df = pd.DataFrame(sentiment_rows)

  for column in sentiment_df.columns:
    working_df[column] = sentiment_df[column]

  working_df["sentiment_score"] = pd.to_numeric(
    working_df["sentiment_score"],
    errors="coerce",
  ).fillna(0.0).astype("float64")

  probability_columns = [
    "sentiment_confidence",
    "sentiment_negative_prob",
    "sentiment_neutral_prob",
    "sentiment_positive_prob",
  ]
  for column in probability_columns:
    working_df[column] = pd.to_numeric(
      working_df[column],
      errors="coerce",
    ).fillna(0.0).astype("float64")

  int_columns = [
    "comment_like_count",
    "reply_count",
    "positive_hits",
    "negative_hits",
    "token_count",
  ]
  for column in int_columns:
    working_df[column] = pd.to_numeric(
      working_df[column],
      errors="coerce",
    ).fillna(0).astype("int64")

  ordered_df = working_df[OUTPUT_COLUMNS].copy()
  ordered_df = ordered_df.drop_duplicates(subset=["comment_id"], keep="last")
  return ordered_df.reset_index(drop=True)