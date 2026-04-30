#!/usr/bin/env bash

set -euo pipefail

COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.mongo-sharded.yml}"

echo "Starting MongoDB sharded cluster containers..."
docker compose -f "$COMPOSE_FILE" up -d

wait_for_mongo() {
  local service_name="$1"
  echo "Waiting for ${service_name}..."

  until docker compose -f "$COMPOSE_FILE" exec -T "$service_name" mongosh --quiet --eval "db.runCommand({ ping: 1 }).ok" >/dev/null 2>&1; do
    sleep 2
  done

  echo "${service_name} is accepting MongoDB commands."
}

wait_for_primary() {
  local service_name="$1"
  local label="$2"

  echo "Waiting for ${label} primary..."

  until docker compose -f "$COMPOSE_FILE" exec -T "$service_name" mongosh --quiet --eval "db.hello().isWritablePrimary" 2>/dev/null | grep -q "true"; do
    sleep 2
  done

  echo "${label} primary is ready."
}

wait_for_mongo "mongo-config01"
wait_for_mongo "mongo-shard01"
wait_for_mongo "mongo-shard02"

echo "Initializing config server replica set..."
docker compose -f "$COMPOSE_FILE" exec -T mongo-config01 mongosh --quiet --eval '
try {
  rs.status();
  print("configReplSet already initialized.");
} catch (error) {
  print("Creating configReplSet...");
  rs.initiate({
    _id: "configReplSet",
    configsvr: true,
    members: [
      {
        _id: 0,
        host: "mongo-config01:27017"
      }
    ]
  });
}
'

echo "Initializing shard01 replica set..."
docker compose -f "$COMPOSE_FILE" exec -T mongo-shard01 mongosh --quiet --eval '
try {
  rs.status();
  print("shard01ReplSet already initialized.");
} catch (error) {
  print("Creating shard01ReplSet...");
  rs.initiate({
    _id: "shard01ReplSet",
    members: [
      {
        _id: 0,
        host: "mongo-shard01:27017"
      }
    ]
  });
}
'

echo "Initializing shard02 replica set..."
docker compose -f "$COMPOSE_FILE" exec -T mongo-shard02 mongosh --quiet --eval '
try {
  rs.status();
  print("shard02ReplSet already initialized.");
} catch (error) {
  print("Creating shard02ReplSet...");
  rs.initiate({
    _id: "shard02ReplSet",
    members: [
      {
        _id: 0,
        host: "mongo-shard02:27017"
      }
    ]
  });
}
'

wait_for_primary "mongo-config01" "configReplSet"
wait_for_primary "mongo-shard01" "shard01ReplSet"
wait_for_primary "mongo-shard02" "shard02ReplSet"

echo "Waiting for mongos router..."
wait_for_mongo "mongo-router"

echo "Adding shards and preparing SocialPulse database..."
docker compose -f "$COMPOSE_FILE" exec -T mongo-router mongosh --quiet --eval '
function safeRun(label, operation) {
  try {
    print("\n" + label);
    printjson(operation());
  } catch (error) {
    print(label + " skipped or already done: " + error.message);
  }
}

safeRun("Adding shard01", function() {
  return sh.addShard("shard01ReplSet/mongo-shard01:27017");
});

safeRun("Adding shard02", function() {
  return sh.addShard("shard02ReplSet/mongo-shard02:27017");
});

safeRun("Enabling sharding on socialpulse database", function() {
  return sh.enableSharding("socialpulse");
});

const socialpulseDb = db.getSiblingDB("socialpulse");

safeRun("Creating youtube_comments_raw collection", function() {
  return socialpulseDb.createCollection("youtube_comments_raw");
});

safeRun("Creating comment identity index", function() {
  return socialpulseDb.youtube_comments_raw.createIndex(
    {
      run_id: 1,
      query_id: 1,
      video_id: 1,
      comment_id: 1
    },
    {
      name: "idx_comment_identity"
    }
  );
});

safeRun("Creating hashed comment_id index", function() {
  return socialpulseDb.youtube_comments_raw.createIndex(
    {
      comment_id: "hashed"
    },
    {
      name: "idx_comment_id_hashed"
    }
  );
});

safeRun("Sharding youtube_comments_raw collection", function() {
  return sh.shardCollection(
    "socialpulse.youtube_comments_raw",
    {
      comment_id: "hashed"
    }
  );
});

print("\nFinal sharding status:");
sh.status();

print("\nCollection indexes:");
printjson(socialpulseDb.youtube_comments_raw.getIndexes());
'

echo ""
echo "MongoDB sharded cluster is ready."
echo "Application connection URI:"
echo "mongodb://localhost:27017"
