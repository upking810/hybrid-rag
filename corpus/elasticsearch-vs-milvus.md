# Elasticsearch/OpenSearch vs Milvus: Engine Landscape

## Elasticsearch and OpenSearch

Elasticsearch is a distributed search and analytics engine built on Lucene; OpenSearch is its Apache-2.0 fork (created in 2021 after Elastic's license change), now a Linux Foundation project. Both organize data as JSON documents in sharded, replicated indexes, and both grew from full-text engines into hybrid platforms: BM25 over inverted indexes, k-NN vector search (HNSW), aggregations, geo queries, and security/observability workloads. Their strength is being one engine for many retrieval modes — full-text, vector, filters, and aggregations composed in one query, with mature operations (rolling upgrades, snapshots, access control) built over years.

Vector search arrived as an extension to a Lucene-centric architecture, so at extreme vector scale a dedicated engine can be more resource-efficient. But for the very common case — text-heavy corpora needing BM25 + vectors + metadata filtering together — a hybrid engine avoids running and synchronizing two systems.

## Milvus

Milvus is a purpose-built vector database (open source, originally from Zilliz) designed cloud-native: storage and compute separate, with distinct write, query, and index-building paths scaling independently over object storage. It supports multiple index types (HNSW, IVF variants, PQ/SQ compression, DiskANN, GPU indexes) and tunable consistency levels, targeting billions of vectors. Its full-text/sparse capabilities are younger than Lucene's, so Milvus deployments often pair it with a lexical engine or accept weaker keyword search.

The specialized-vs-integrated trade-off runs through the whole market: dedicated engines (Milvus, Qdrant, Weaviate), libraries (Faiss — an ANN library, not a service: no server, replication, or filtering), extensions to existing databases (pgvector), and hybrid search engines (Elasticsearch/OpenSearch). Choosing means weighing vector scale, filtering and full-text needs, operational maturity, and how many systems a team wants to run.

## What a managed retrieval platform adds

Cloud providers (including volcano-engine-style managed search services) wrap these engines as managed services: provisioning, scaling, upgrades, monitoring, security hardening, and increasingly a RAG-ready layer — managed embedding pipelines, hybrid query APIs, and agentic search features — so application teams consume retrieval as a platform capability instead of operating clusters. The engineering work behind such platforms concentrates on multi-tenancy isolation, cost efficiency at scale, and keeping compatibility with upstream open-source APIs.
