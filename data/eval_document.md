# Retrieval-Augmented Generation: Architecture, Components, and Best Practices

## Introduction

Retrieval-Augmented Generation (RAG) is an AI architecture that combines the power of large language models (LLMs) with external knowledge retrieval to produce more accurate and grounded responses. Introduced by Lewis et al. in 2020, RAG addresses a key limitation of static LLMs: their inability to access information beyond their training data. In a RAG system, when a user asks a question, the system first retrieves relevant documents from a knowledge base, then uses those documents as context to generate a response. This approach significantly reduces hallucination and keeps answers factually grounded.

## Core Components of a RAG Pipeline

A RAG pipeline consists of three primary components:

**The Retriever** is responsible for finding relevant documents from a corpus. The retriever typically converts both documents and queries into dense vector embeddings and performs approximate nearest-neighbor (ANN) search to find the most semantically similar documents. The quality of the retriever is the most important factor in overall RAG system performance.

**The Vector Database** stores precomputed document embeddings and enables fast similarity search. Popular vector databases include pgvector (a PostgreSQL extension), Pinecone, Weaviate, Chroma, and FAISS. These databases use specialized indexing structures such as HNSW (Hierarchical Navigable Small World) graphs for efficient retrieval at scale.

**The Generator** is a large language model that takes the retrieved documents as context and generates a final response. The generator synthesizes information from multiple retrieved passages to produce coherent, grounded answers. The generator cannot access information beyond what the retriever provides in the context window.

## Document Chunking Strategies

Before documents can be embedded and stored, they must be split into smaller segments called chunks. Chunking strategy significantly impacts retrieval quality because chunks that are too large introduce noise while chunks that are too small lose important context.

### Recursive Character Text Splitting

The most common approach is recursive character text splitting, which attempts to split text at natural boundaries: first at double newlines (paragraph boundaries), then single newlines, then sentences, then words, and finally individual characters as a last resort. A chunk size of 512 to 1024 characters is recommended for most use cases, balancing between too-broad context (which introduces noise) and too-narrow context (which loses important surrounding information). Larger documents with long structured sections may benefit from chunk sizes up to 2048 characters.

### Semantic Chunking

Semantic chunking groups paragraphs by topic using embedding similarity. Adjacent paragraphs with cosine similarity above a threshold (typically 0.5 to 0.7) are merged into the same chunk. This produces topically coherent chunks but requires running an embedding model during ingestion, making it slower than recursive splitting.

### Chunk Overlap

Chunk overlap ensures that information at chunk boundaries is not lost. A typical overlap ratio is 10 to 15 percent of the chunk size. For example, a 1024-character chunk would have approximately 100 to 150 characters of overlap with adjacent chunks. The overlap is applied symmetrically: the end of one chunk reappears at the beginning of the next, preserving context that would otherwise be split between two separate chunks.

## Embedding Models

Embedding models convert text into dense vector representations that capture semantic meaning. The choice of embedding model directly affects retrieval quality because it determines how well semantic similarity between queries and documents is captured.

### Bi-Encoders

Bi-encoders (also called dual encoders) independently encode the query and documents into fixed-size vectors. They are fast for retrieval because document embeddings can be precomputed and stored in the vector database. Popular bi-encoder models include BGE-M3 (multilingual, 1024 dimensions), all-MiniLM-L6-v2 (English-focused, 384 dimensions), and text-embedding-3-small (OpenAI, 1536 dimensions). The main limitation of bi-encoders is that the query and document are encoded independently, so the model cannot capture fine-grained interactions between them.

### Cross-Encoders

Cross-encoders process the query and document together in a single forward pass through the model, enabling richer token-level interaction between query and document. They achieve higher relevance scoring accuracy than bi-encoders but cannot precompute document embeddings, making them too slow for first-stage retrieval over large corpora. Cross-encoders are typically used in a re-ranking stage to re-score the top candidates from a bi-encoder retrieval.

### Late Interaction Models

Models such as ColBERT use late interaction: they precompute token-level embeddings for documents (providing precomputation benefits) while still allowing token-level matching at query time. This provides a middle ground between bi-encoder speed and cross-encoder accuracy, at the cost of significantly higher storage requirements for the per-token embeddings.

## HNSW Indexing

Hierarchical Navigable Small World (HNSW) is a graph-based approximate nearest-neighbor algorithm used by most vector databases. HNSW builds a multi-layer graph where higher layers provide coarse long-range navigation and lower layers enable fine-grained local search. During query time, the search starts at the top layer and greedily descends to progressively denser layers until it reaches the target node.

Key HNSW parameters include:

- **m**: The number of bidirectional connections per node (default: 16). Higher values improve recall at the cost of increased memory usage and build time.
- **ef_construction**: The size of the dynamic candidate list during index construction (default: 64). Higher values produce a higher-quality graph at the cost of slower index build time.
- **ef_search**: The size of the dynamic candidate list during query execution. Higher values improve recall at the cost of increased query latency.

The trade-off between recall and latency is controlled primarily by ef_search, which can be tuned at query time without rebuilding the index.

## RAG Evaluation Metrics — RAGAS Framework

The RAGAS (Retrieval-Augmented Generation Assessment) framework provides automated metrics for evaluating RAG pipelines without requiring manual human annotation. RAGAS uses an LLM-as-judge approach where a language model evaluates different aspects of the RAG output.

### Faithfulness

Faithfulness measures whether the generated answer is factually consistent with the retrieved context. The metric decomposes the answer into individual factual statements and checks whether each statement can be inferred from the provided context documents. Faithfulness score ranges from 0 to 1, where 1 means every statement in the answer is supported by the context and 0 means no statements are supported. A low faithfulness score indicates that the model is generating information not present in the retrieved documents, which is known as hallucination. Faithfulness is the most critical metric for production RAG systems because hallucinated answers can be dangerous.

### Answer Relevancy

Answer relevancy measures how pertinent the generated answer is to the user's original question. It is computed by having a language model generate several alternative questions from the answer, then measuring the average cosine similarity between these reverse-engineered questions and the original question. A high answer relevancy score indicates the answer directly addresses the question without containing excessive irrelevant or tangential information. A low score may indicate the answer addresses a different question than the one asked.

### Context Precision

Context precision (also called context relevancy) measures whether the retrieved context is relevant to the question and the expected answer. It evaluates the signal-to-noise ratio of the retrieved passages: a high context precision score means most retrieved chunks are relevant to answering the question, while a low score means many irrelevant chunks were retrieved alongside the relevant ones. Context precision is important because irrelevant context dilutes the useful signal available to the generator and can mislead it into producing off-topic answers.

### Context Recall

Context recall measures the fraction of the reference answer's information that is covered by the retrieved context. Unlike context precision, which measures the quality of what was retrieved, context recall measures completeness — whether anything important was missed. A high context recall score indicates the retriever successfully found all passages needed to answer the question fully.

## Two-Stage Retrieval with Re-Ranking

Two-stage retrieval improves retrieval precision by adding a re-ranking step after the initial bi-encoder retrieval:

**Stage 1 — Recall**: Use a fast bi-encoder to retrieve a large candidate set, typically the top 20 to 50 documents. This stage prioritizes recall (finding all relevant documents) over precision.

**Stage 2 — Precision**: Use a slower but more accurate cross-encoder to re-score and re-rank the candidates, then select the final top-k (typically top 3 to 5). Because the cross-encoder only needs to score a small candidate set rather than the entire corpus, it remains tractable even though it cannot precompute document embeddings.

This combination achieves better precision than bi-encoder alone while remaining computationally feasible. The cross-encoder model ms-marco-MiniLM-L-6-v2 is widely used for re-ranking due to its excellent accuracy-to-speed balance and relatively small size (approximately 80 MB).

## Limitations of RAG Systems

Despite their advantages over purely parametric LLMs, RAG systems have several inherent limitations:

**Retrieval failures**: If the retriever does not find the relevant documents, the generator will lack sufficient context to answer correctly. This "garbage in, garbage out" problem means retrieval quality is a hard ceiling on answer quality — the generator cannot compensate for a failed retrieval.

**Context window constraints**: LLMs have a maximum context length (typically 4K to 128K tokens). When the total size of retrieved documents exceeds this limit, some context must be truncated, potentially losing critical information.

**Multi-hop reasoning**: Questions that require combining information from multiple disconnected documents are challenging for RAG. The retriever may not retrieve all necessary documents in a single query, and the generator may struggle to reason across multiple retrieved passages.

**Latency**: Adding a retrieval step, especially with re-ranking, increases end-to-end response latency compared to direct LLM inference. Real-time applications with strict latency budgets may find this trade-off unacceptable.

**Out-of-date knowledge**: The knowledge base reflects a snapshot in time. If it is not regularly updated with new documents, the RAG system will provide outdated information even when more recent knowledge exists.

**Chunking artifacts**: Poor chunking decisions can split important context across multiple chunks so that no single retrieved chunk contains enough information to answer a question. This is particularly problematic for tables, lists, and structured data that span multiple pages.
