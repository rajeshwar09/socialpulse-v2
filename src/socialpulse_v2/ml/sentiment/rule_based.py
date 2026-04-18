from __future__ import annotations

import re
from dataclasses import dataclass


POSITIVE_WORDS = {
  "good", "great", "awesome", "amazing", "best", "love", "loved", "excellent",
  "fantastic", "super", "nice", "cool", "worth", "perfect", "impressive",
  "favorite", "favourite", "strong", "smooth", "premium", "useful", "helpful",
  "beautiful", "fast", "powerful", "happy", "wow", "fire"
}

NEGATIVE_WORDS = {
  "bad", "worst", "waste", "poor", "hate", "hated", "boring", "fake", "useless",
  "slow", "lag", "laggy", "issue", "problem", "overpriced", "expensive", "cheap",
  "disappointing", "disappointed", "terrible", "awful", "broken", "bug", "bugs",
  "cringe", "mid", "weak", "class", "trash", "scam"
}

POSITIVE_EMOJIS = {"🔥", "😍", "❤️", "❤", "💕", "💯", "👏", "🙌", "🥳", "😊", "😁", "🤩", "👌", "👍", "🎉"}
NEGATIVE_EMOJIS = {"😡", "🤮", "👎", "💩", "😒", "😞", "😠", "😭", "🤦", "🙄", "😤"}

NEGATION_WORDS = {"not", "no", "never", "hardly", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't"}


@dataclass(frozen=True)
class SentimentResult:
  sentiment_score: float
  sentiment_label: str
  positive_hits: int
  negative_hits: int
  token_count: int


def normalize_text(text: str) -> str:
  cleaned = text.lower().strip()
  cleaned = cleaned.replace("’", "'")
  return cleaned


def tokenize(text: str) -> list[str]:
  return re.findall(r"[a-zA-Z']+", text.lower())


def classify_sentiment(score: float) -> str:
  if score > 0.2:
    return "positive"
  if score < -0.2:
    return "negative"
  return "neutral"


def score_text(text: str) -> SentimentResult:
  normalized = normalize_text(text)
  tokens = tokenize(normalized)

  positive_hits = 0
  negative_hits = 0

  for idx, token in enumerate(tokens):
    prev_token = tokens[idx - 1] if idx > 0 else ""

    if token in POSITIVE_WORDS:
      if prev_token in NEGATION_WORDS:
        negative_hits += 1
      else:
        positive_hits += 1

    if token in NEGATIVE_WORDS:
      if prev_token in NEGATION_WORDS:
        positive_hits += 1
      else:
        negative_hits += 1

  positive_hits += sum(1 for emoji in POSITIVE_EMOJIS if emoji in text)
  negative_hits += sum(1 for emoji in NEGATIVE_EMOJIS if emoji in text)

  token_count = len(tokens)
  denominator = max(token_count, 1)
  score = (positive_hits - negative_hits) / denominator

  return SentimentResult(
    sentiment_score=round(score, 4),
    sentiment_label=classify_sentiment(score),
    positive_hits=positive_hits,
    negative_hits=negative_hits,
    token_count=token_count,
  )
