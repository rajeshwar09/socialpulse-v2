from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 256

LABEL_ALIASES = {
  "label_0": "negative",
  "negative": "negative",
  "neg": "negative",
  "1_star": "negative",
  "1_stars": "negative",
  "2_stars": "negative",
  "label_1": "neutral",
  "neutral": "neutral",
  "neu": "neutral",
  "3_stars": "neutral",
  "label_2": "positive",
  "positive": "positive",
  "pos": "positive",
  "4_stars": "positive",
  "5_stars": "positive",
}


@dataclass(frozen=True)
class TransformerSentimentScore:
  sentiment_score: float
  sentiment_label: str
  sentiment_confidence: float
  sentiment_negative_prob: float
  sentiment_neutral_prob: float
  sentiment_positive_prob: float
  token_count: int
  model_name: str


def _normalize_label(label: str) -> str:
  cleaned = label.strip().lower().replace("-", "_").replace(" ", "_")
  return LABEL_ALIASES.get(cleaned, cleaned)


def _token_count(text: str) -> int:
  return len(re.findall(r"[A-Za-z0-9']+", text))


def _preprocess_text(text: str) -> str:
  normalized = " ".join(str(text).strip().split())
  if not normalized:
    return ""

  tokens: list[str] = []
  for token in normalized.split(" "):
    if token.startswith("@") and len(token) > 1:
      tokens.append("@user")
    elif token.startswith("http"):
      tokens.append("http")
    else:
      tokens.append(token)
  return " ".join(tokens)


def _resolve_device() -> torch.device:
  requested = os.getenv("SOCIALPULSE_TRANSFORMER_DEVICE", "cpu").strip().lower()

  if requested == "cuda" and torch.cuda.is_available():
    return torch.device("cuda")

  if requested == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    return torch.device("mps")

  return torch.device("cpu")


class TransformerSentimentInference:
  def __init__(
    self,
    model_name: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
  ) -> None:
    self.model_name = model_name or os.getenv("SOCIALPULSE_TRANSFORMER_MODEL", DEFAULT_MODEL_NAME)
    self.batch_size = batch_size or int(os.getenv("SOCIALPULSE_TRANSFORMER_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    self.max_length = max_length or int(os.getenv("SOCIALPULSE_TRANSFORMER_MAX_LENGTH", str(DEFAULT_MAX_LENGTH)))
    self.device = _resolve_device()

    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
    self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
    self.model.to(self.device)
    self.model.eval()

  def score_texts(self, texts: Iterable[str]) -> list[TransformerSentimentScore]:
    prepared_texts = [_preprocess_text(text) for text in texts]
    results: list[TransformerSentimentScore] = []

    for start in range(0, len(prepared_texts), self.batch_size):
      batch = prepared_texts[start:start + self.batch_size]

      blank_positions = [index for index, text in enumerate(batch) if text == ""]
      batch_results: list[TransformerSentimentScore | None] = [None] * len(batch)

      non_blank_positions = [index for index, text in enumerate(batch) if text != ""]
      non_blank_texts = [batch[index] for index in non_blank_positions]

      if non_blank_texts:
        encoded = self.tokenizer(
          non_blank_texts,
          padding=True,
          truncation=True,
          max_length=self.max_length,
          return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
          logits = self.model(**encoded).logits
          probabilities = torch.softmax(logits, dim=-1).detach().cpu().tolist()

        for local_index, probs in enumerate(probabilities):
          label_map: dict[str, float] = {}
          for class_index, raw_score in enumerate(probs):
            raw_label = str(self.model.config.id2label.get(class_index, f"LABEL_{class_index}"))
            normalized_label = _normalize_label(raw_label)
            label_map[normalized_label] = float(raw_score)

          negative_prob = float(label_map.get("negative", 0.0))
          neutral_prob = float(label_map.get("neutral", 0.0))
          positive_prob = float(label_map.get("positive", 0.0))

          label_candidates = {
            "negative": negative_prob,
            "neutral": neutral_prob,
            "positive": positive_prob,
          }
          sentiment_label = max(label_candidates, key=label_candidates.get) # type: ignore
          sentiment_confidence = float(label_candidates[sentiment_label])
          sentiment_score = round(positive_prob - negative_prob, 6)

          original_text = non_blank_texts[local_index]
          batch_results[non_blank_positions[local_index]] = TransformerSentimentScore(
            sentiment_score=sentiment_score,
            sentiment_label=sentiment_label,
            sentiment_confidence=round(sentiment_confidence, 6),
            sentiment_negative_prob=round(negative_prob, 6),
            sentiment_neutral_prob=round(neutral_prob, 6),
            sentiment_positive_prob=round(positive_prob, 6),
            token_count=_token_count(original_text),
            model_name=self.model_name,
          )

      for blank_index in blank_positions:
        batch_results[blank_index] = TransformerSentimentScore(
          sentiment_score=0.0,
          sentiment_label="neutral",
          sentiment_confidence=1.0,
          sentiment_negative_prob=0.0,
          sentiment_neutral_prob=1.0,
          sentiment_positive_prob=0.0,
          token_count=0,
          model_name=self.model_name,
        )

      results.extend([result for result in batch_results if result is not None])

    return results


@lru_cache(maxsize=1)
def get_transformer_inference() -> TransformerSentimentInference:
  return TransformerSentimentInference()


def score_texts(texts: Iterable[str]) -> list[TransformerSentimentScore]:
  return get_transformer_inference().score_texts(texts)


def score_text(text: str) -> TransformerSentimentScore:
  return score_texts([text])[0]