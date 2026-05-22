from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from py_rag_engine.chunker import SemanticChunker
from py_rag_engine.embedder import VectorClient
from py_rag_engine.vector_math import cosine_similarity

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
EMBED_MODEL = "text-embedding-bge-m3"

PARAGRAPHS = [
    "The history of Python begins in the late 1980s when Guido van Rossum started working on a successor to the ABC language.",
    "Python emphasizes code readability with its notable use of significant indentation and a clean, expressive syntax.",
    "The Roman Empire was one of the largest civilizations of antiquity, controlling vast territories across Europe, Africa, and Asia.",
    "Roman engineers built aqueducts, roads, and amphitheaters that survived for centuries after the empire fell.",
    "Quantum computing uses qubits instead of classical bits, leveraging superposition and entanglement to perform certain calculations exponentially faster than classical machines.",
]


async def main() -> None:
    openai_client = AsyncOpenAI(base_url=LM_STUDIO_URL, api_key="lm-studio")
    embedder = VectorClient(
        provider="openai", model=EMBED_MODEL, client=openai_client, batch_size=8
    )

    vectors = await embedder.get_embeddings(PARAGRAPHS)

    print("Adjacent cosine distances (1 - similarity):")
    for i in range(len(vectors) - 1):
        sim = cosine_similarity(vectors[i], vectors[i + 1])
        dist = 1.0 - sim
        topic_a = ["python", "python", "rome", "rome", "quantum"][i]
        topic_b = ["python", "python", "rome", "rome", "quantum"][i + 1]
        marker = "SAME" if topic_a == topic_b else "SHIFT"
        print(f"  [{i}->{i + 1}] dist={dist:.4f} ({topic_a} -> {topic_b}) [{marker}]")

    print("\nChunking with threshold tuned between intra/inter-topic distances:")
    chunker = SemanticChunker(
        embedder=embedder, distance_threshold=0.55, max_paragraphs_per_chunk=10
    )
    text = "\n\n".join(PARAGRAPHS)
    chunks = await chunker.chunk(text, page=1, source="probe.md")
    print(f"  -> {len(chunks)} chunks")
    for c in chunks:
        n_paras = c.text.count("\n\n") + 1
        print(f"    idx={c.metadata.chunk_index} paragraphs={n_paras} :: {c.text[:60]!r}...")


if __name__ == "__main__":
    asyncio.run(main())
