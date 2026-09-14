# IVF and Product Quantization

## IVF: inverted file index for vectors

IVF (inverted file) is the other major ANN family, popularized by Faiss. Offline, cluster all vectors into nlist buckets with k-means; each bucket keeps the list of vectors assigned to it — an "inverted list", by analogy to a term's postings list. At query time, find the nprobe centroids closest to the query and scan only those buckets instead of the whole corpus.

nprobe is the recall/latency knob: probing more buckets finds more true neighbors but costs more distance computations. IVF's weakness is boundary effects — a true neighbor sitting just across a cluster boundary is missed unless its bucket is probed. Compared to HNSW, IVF is cheaper to build and more memory-friendly, but usually needs more tuning to match HNSW's recall at the same latency.

## Product quantization: compressing vectors

Product quantization (PQ) attacks memory rather than speed. Split each vector into m sub-vectors; for each sub-space, learn a small codebook (typically 256 centroids) with k-means; store each sub-vector as the 1-byte id of its nearest centroid. A 768-dimensional float32 vector (3 KB) becomes m bytes — e.g. 96 bytes at m=96, a 32x compression.

Distances are computed approximately against the codebooks (asymmetric distance computation, using per-query lookup tables), so PQ trades accuracy for a dramatic memory reduction. IVF-PQ combines both: cluster for pruning, quantize for compression — this is the workhorse configuration for billion-scale datasets. A common refinement is re-ranking the top PQ candidates with full-precision vectors kept on disk.

## Scalar quantization and disk-based indexes

Simpler than PQ, scalar quantization (SQ) stores each dimension as int8 instead of float32 — 4x compression with minimal recall loss, widely used as a default (e.g. in Milvus and OpenSearch). At the extreme, disk-based designs like DiskANN keep compressed vectors in RAM for navigation and full vectors on NVMe SSD, enabling billion-scale search on a single machine. Choosing among HNSW, IVF, PQ variants, and disk indexes is a cost-engineering exercise: recall target, latency budget, memory budget, update rate.
