from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from transformers import pipeline


DEFAULT_MODEL_NAME = os.getenv(
  "SOCIALPULSE_SENTIMENT_MODEL",
  "cardiffnlp/twitter-roberta-base-sentiment-latest",
)


@dataclass(frozen=True)
class TransformerSentimentResult:
  sentiment_score: float
  sentiment_label: str
  sentiment_confidence: float
  positive_probability: float
  neutral_probability: float
  negative_probability: float
  token_count: int
  sentiment_model: str


def _normalize_label(label: str) -> str:
  cleaned = label.strip().upper()

  mapping = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
    "NEGATIVE": "negative",
    "NEUTRAL": "neutral",
    "POSITIVE": "positive",
  }

  if cleaned in mapping:
    return mapping[cleaned]

  if "NEG" in cleaned:
    return "negative"
  if "NEU" in cleaned:
    return "neutral"
  if "POS" in cleaned:
    return "positive"

  return "neutral"


def _count_tokens(text: str) -> int:
  return len(re.findall(r"\S+", text))


@lru_cache(maxsize=4)
def _build_classifier(model_name: str, device: int):
  return pipeline(
    task="text-classification",
    model=model_name,
    tokenizer=model_name,
    device=device,
  )


def _empty_result(model_name: str) -> TransformerSentimentResult:
  return TransformerSentimentResult(
    sentiment_score=0.0,
    sentiment_label="neutral",
    sentiment_confidence=1.0,
    positive_probability=0.0,
    neutral_probability=1.0,
    negative_probability=0.0,
    token_count=0,
    sentiment_model=model_name,
  )


def _scores_to_result(
  text: str,
  raw_scores: list[dict],
  model_name: str,
) -> TransformerSentimentResult:
  probs = {
    "positive": 0.0,
    "neutral": 0.0,
    "negative": 0.0,
  }

  for item in raw_scores:
    mapped_label = _normalize_label(str(item.get("label", "")))
    probs[mapped_label] = float(item.get("score", 0.0))

  sentiment_label = max(probs, key=probs.get)
  sentiment_score = round(probs["positive"] - probs["negative"], 4)
  sentiment_confidence = round(max(probs.values()), 4)

  return TransformerSentimentResult(
    sentiment_score=sentiment_score,
    sentiment_label=sentiment_label,
    sentiment_confidence=sentiment_confidence,
    positive_probability=round(probs["positive"], 4),
    neutral_probability=round(probs["neutral"], 4),
    negative_probability=round(probs["negative"], 4),
    token_count=_count_tokens(text),
    sentiment_model=model_name,
  )


def score_texts_transformer(
  texts: list[str],
  model_name: str | None = None,
  batch_size: int | None = None,
  max_length: int | None = None,
) -> list[TransformerSentimentResult]:
  resolved_model = model_name or DEFAULT_MODEL_NAME
  resolved_batch_size = batch_size or int(os.getenv("SOCIALPULSE_SENTIMENT_BATCH_SIZE", "32"))
  resolved_max_length = max_length or int(os.getenv("SOCIALPULSE_SENTIMENT_MAX_LENGTH", "256"))
  resolved_device = int(os.getenv("SOCIALPULSE_SENTIMENT_DEVICE", "-1"))

  classifier = _build_classifier(resolved_model, resolved_device)

  safe_texts = ["" if text is None else str(text) for text in texts]
  results = [_empty_result(resolved_model) for _ in safe_texts]

  active_indexes: list[int] = []
  active_texts: list[str] = []

  for idx, text in enumerate(safe_texts):
    if text.strip():
      active_indexes.append(idx)
      active_texts.append(text)

  if not active_texts:
    return results

  raw_outputs = classifier(
    active_texts,
    top_k=None,
    truncation=True,
    max_length=resolved_max_length,
    batch_size=resolved_batch_size,
  )

  for idx, raw_scores in zip(active_indexes, raw_outputs):
    results[idx] = _scores_to_result(
      text=safe_texts[idx],
      raw_scores=raw_scores,
      model_name=resolved_model,
    )

  return results