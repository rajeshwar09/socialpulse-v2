#!/usr/bin/env bash

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.mongo-sharded.yml}"

echo ""
echo "Docker containers:"
docker compose -f "$COMPOSE_FILE" ps

echo ""
echo "MongoDB sharded cluster check:"
docker compose -f "$COMPOSE_FILE" exec -T mongo-router mongosh --quiet --eval '
print("MongoDB version: " + db.version());

print("\n--- Sharding status ---");
sh.status();

const socialpulseDb = db.getSiblingDB("socialpulse");

print("\n--- SocialPulse collections ---");
printjson(socialpulseDb.getCollectionNames());

print("\n--- youtube_comments_raw document count ---");
print(socialpulseDb.youtube_comments_raw.countDocuments({}));

print("\n--- youtube_comments_raw indexes ---");
printjson(socialpulseDb.youtube_comments_raw.getIndexes());

print("\n--- youtube_comments_raw shard distribution ---");
socialpulseDb.youtube_comments_raw.getShardDistribution();
'
