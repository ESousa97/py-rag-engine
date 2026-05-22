from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from py_rag_engine.chunker import SemanticChunker
from py_rag_engine.embedder import VectorClient
from py_rag_engine.embeddings.hashing import content_sha256

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
EMBED_MODEL = "text-embedding-bge-m3"

DOCUMENT = (
    "The history of Python begins in the late 1980s when Guido van Rossum "
    "started working on a successor to the ABC language.\n\n"
    "Python emphasizes code readability with its notable use of significant "
    "indentation and a clean, expressive syntax.\n\n"
    "The Roman Empire was one of the largest civilizations of antiquity, "
    "controlling vast territories across Europe, Africa, and Asia.\n\n"
    "Roman engineers built aqueducts, roads, and amphitheaters that survived "
    "for centuries after the empire fell.\n\n"
    "Quantum computing uses qubits instead of classical bits, leveraging "
    "superposition and entanglement to perform certain calculations exponentially "
    "faster than classical machines."
)


async def main() -> None:
    openai_client = AsyncOpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")

    embedder = VectorClient(
        provider="openai",
        model=EMBED_MODEL,
        client=openai_client,
        batch_size=8,
    )

    print("=== Test 1: VectorClient.get_embeddings ===")
    vectors = await embedder.get_embeddings(["hello world", "olá mundo"])
    assert len(vectors) == 2, "Expected 2 vectors"
    assert all(len(v) > 0 for v in vectors), "Vectors must be non-empty"
    print(f"  OK: {len(vectors)} vectors of dim={len(vectors[0])}")

    print("\n=== Test 2: VectorClient.get_embeddings([]) ===")
    empty = await embedder.get_embeddings([])
    assert empty == []
    print("  OK: empty input returns []")

    print("\n=== Test 3: SemanticChunker.chunk on multi-topic document ===")
    chunker = SemanticChunker(
        embedder=embedder,
        distance_threshold=0.18,
        max_paragraphs_per_chunk=10,
    )
    chunks = await chunker.chunk(DOCUMENT, page=1, source="practical_test.md")
    print(f"  Generated {len(chunks)} chunks from 5 paragraphs:")
    for chunk in chunks:
        preview = chunk.text.replace("\n\n", " | ")[:80]
        print(
            f"    idx={chunk.metadata.chunk_index} "
            f"page={chunk.metadata.page} "
            f"hash={chunk.content_hash[:10]}... "
            f"len={len(chunk.text)} :: {preview!r}"
        )

    assert len(chunks) >= 2, "Expected at least 2 semantic groups in a multi-topic doc"

    print("\n=== Test 4: metadata integrity ===")
    for i, chunk in enumerate(chunks):
        assert chunk.metadata.page == 1, f"page mismatch on chunk {i}"
        assert chunk.metadata.source == "practical_test.md"
        assert chunk.metadata.chunk_index == i
        assert chunk.content_hash == content_sha256(chunk.text)
    print("  OK: pages, source, chunk_index, and SHA-256 hashes all correct")

    print("\n=== Test 5: idempotency (re-run produces identical hashes) ===")
    chunks_again = await chunker.chunk(DOCUMENT, page=1, source="practical_test.md")
    hashes_a = [c.content_hash for c in chunks]
    hashes_b = [c.content_hash for c in chunks_again]
    assert hashes_a == hashes_b, "Hashes must be identical across runs"
    print(f"  OK: {len(hashes_a)} hashes identical on second run")

    print("\n=== Test 6: chunk_index_start offset ===")
    offset_chunks = await chunker.chunk(
        DOCUMENT, page=2, source="practical_test.md", chunk_index_start=100
    )
    assert offset_chunks[0].metadata.chunk_index == 100
    assert offset_chunks[0].metadata.page == 2
    print(f"  OK: first chunk_index={offset_chunks[0].metadata.chunk_index}, page=2")

    print("\n=== Test 7: empty / single-paragraph input ===")
    empty_chunks = await chunker.chunk("", page=1, source="x")
    assert empty_chunks == []
    single_chunks = await chunker.chunk("just one paragraph.", page=1, source="x")
    assert len(single_chunks) == 1
    assert single_chunks[0].text == "just one paragraph."
    print("  OK: empty -> []; single paragraph -> 1 chunk without embeddings call")

    print("\nALL PRACTICAL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
