from pydantic import BaseModel, Field
from typing import Optional


class IngestionRunRecord(BaseModel):
  run_id: str
  run_date: str
  source_name: str
  ingestion_mode: str
  query_id: Optional[str] = None
  query_text: Optional[str] = None
  topic: Optional[str] = None
  genre: Optional[str] = None
  status: str
  records_fetched: int = 0
  records_written: int = 0
  error_message: Optional[str] = None
  created_at: str


class YouTubeCommentRawRecord(BaseModel):
  source_run_id: str
  ingestion_type: str
  query_id: Optional[str] = None
  query_text: Optional[str] = None
  topic: Optional[str] = None
  genre: Optional[str] = None
  video_id: Optional[str] = None
  channel_id: Optional[str] = None
  comment_id: str
  parent_comment_id: Optional[str] = None
  author_name: Optional[str] = None
  text: str
  like_count: Optional[int] = None
  reply_count: Optional[int] = None
  published_at: Optional[str] = None
  fetched_at: str
  language_target: Optional[str] = None


class YouTubeCommentCleanRecord(BaseModel):
  comment_id: str
  parent_comment_id: Optional[str] = None
  video_id: Optional[str] = None
  channel_id: Optional[str] = None
  query_id: Optional[str] = None
  query_text: Optional[str] = None
  topic: Optional[str] = None
  genre: Optional[str] = None
  clean_text: str
  text_length: int
  like_count: int = 0
  reply_count: int = 0
  published_at: Optional[str] = None
  fetched_at: str
  source_run_id: str
  ingestion_type: str


class CollectionDailySummaryRecord(BaseModel):
  run_date: str
  source_name: str
  ingestion_mode: str
  topic: Optional[str] = None
  genre: Optional[str] = None
  query_id: Optional[str] = None
  query_text: Optional[str] = None
  total_records_fetched: int = 0
  total_records_written: int = 0
  created_at: str