from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import BulkWriteError

from socialpulse_v2.core.settings import settings


@dataclass(frozen=True)
class MongoCollectionConfig:
  uri: str
  database_name: str
  collection_name: str
  server_selection_timeout_ms: int
  connect_timeout_ms: int
  socket_timeout_ms: int


def get_mongo_config() -> MongoCollectionConfig:
  return MongoCollectionConfig(
    uri=settings.mongo_uri,
    database_name=settings.mongo_database,
    collection_name=settings.mongo_youtube_comments_collection,
    server_selection_timeout_ms=settings.mongo_server_selection_timeout_ms,
    connect_timeout_ms=settings.mongo_connect_timeout_ms,
    socket_timeout_ms=settings.mongo_socket_timeout_ms,
  )


def get_mongo_client(config: MongoCollectionConfig | None = None) -> MongoClient:
  active_config = config or get_mongo_config()

  return MongoClient(
    active_config.uri,
    serverSelectionTimeoutMS=active_config.server_selection_timeout_ms,
    connectTimeoutMS=active_config.connect_timeout_ms,
    socketTimeoutMS=active_config.socket_timeout_ms,
  )


def get_socialpulse_database(client: MongoClient, config: MongoCollectionConfig | None = None) -> Database:
  active_config = config or get_mongo_config()
  return client[active_config.database_name]


def get_youtube_comments_collection(
  client: MongoClient,
  config: MongoCollectionConfig | None = None,
) -> Collection:
  active_config = config or get_mongo_config()
  database = get_socialpulse_database(client, active_config)
  return database[active_config.collection_name]


def ping_mongo() -> dict[str, Any]:
  config = get_mongo_config()

  with get_mongo_client(config) as client:
    ping_result = client.admin.command("ping")
    server_info = client.server_info()
    database = get_socialpulse_database(client, config)
    collection = get_youtube_comments_collection(client, config)

    return {
      "ok": ping_result.get("ok") == 1.0,
      "mongo_version": server_info.get("version"),
      "database": config.database_name,
      "collection": config.collection_name,
      "collection_names": database.list_collection_names(),
      "document_count": collection.count_documents({}),
      "indexes": list(collection.list_indexes()),
    }


def ensure_youtube_comment_indexes(collection: Collection) -> None:
  collection.create_index(
    [
      ("run_id", 1),
      ("query_id", 1),
      ("video_id", 1),
      ("comment_id", 1),
    ],
    name="idx_comment_identity",
  )

  collection.create_index(
    [
      ("comment_id", "hashed"),
    ],
    name="idx_comment_id_hashed",
  )

  collection.create_index(
    [
      ("collection_date", 1),
      ("topic", 1),
      ("query_id", 1),
    ],
    name="idx_collection_topic_query",
  )


def upsert_youtube_comment_documents(
  collection: Collection,
  documents: list[dict[str, Any]],
) -> dict[str, int]:
  if not documents:
    return {
      "input_documents": 0,
      "matched_documents": 0,
      "modified_documents": 0,
      "upserted_documents": 0,
    }

  operations: list[UpdateOne] = []

  for document in documents:
    run_id = str(document.get("run_id", ""))
    query_id = str(document.get("query_id", ""))
    video_id = str(document.get("video_id", ""))
    comment_id = str(document.get("comment_id", ""))

    if not comment_id:
      continue

    operations.append(
      UpdateOne(
        {
          "run_id": run_id,
          "query_id": query_id,
          "video_id": video_id,
          "comment_id": comment_id,
        },
        {
          "$set": document,
        },
        upsert=True,
      )
    )

  if not operations:
    return {
      "input_documents": len(documents),
      "matched_documents": 0,
      "modified_documents": 0,
      "upserted_documents": 0,
    }

  try:
    result = collection.bulk_write(operations, ordered=False)
  except BulkWriteError as error:
    details = error.details
    raise RuntimeError(f"MongoDB bulk write failed: {details}") from error

  return {
    "input_documents": len(documents),
    "matched_documents": result.matched_count,
    "modified_documents": result.modified_count,
    "upserted_documents": len(result.upserted_ids),
  }
