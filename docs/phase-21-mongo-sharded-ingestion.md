# Phase 21 — MongoDB Sharded Ingestion Layer

## Goal

The goal of this phase was to add MongoDB as a scalable raw ingestion layer between the YouTube raw data collection and the lakehouse bronze layer.

Earlier, the project was directly using local raw JSON files for downstream lakehouse processing. In this phase, a local sharded MongoDB cluster was added so that raw YouTube comments can first be stored in a database layer and then loaded into the bronze Delta Lake table.

This makes the architecture more realistic because MongoDB acts as a distributed operational/raw storage layer before analytical processing.

---

## Branch

```text
21-mongo-sharded-ingestion
