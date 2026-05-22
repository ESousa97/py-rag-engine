from __future__ import annotations

import asyncio

from openai import AsyncOpenAI

from py_rag_engine import SemanticChunker, VectorClient, calibrate_distance_threshold
from py_rag_engine.chunker import DEFAULT_DISTANCE_THRESHOLD

LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
EMBED_MODEL = "text-embedding-bge-m3"

CALIBRATION_SAMPLE = (
    "Python is a high-level interpreted language created by Guido van Rossum.\n\n"
    "Python emphasizes readability through significant indentation and minimal syntax.\n\n"
    "Python supports multiple paradigms: procedural, object-oriented, and functional.\n\n"
    "The standard library ships with batteries-included modules for nearly every domain.\n\n"
    "Pip installs third-party packages from PyPI for extra functionality.\n\n"
    "Virtual environments isolate dependencies for each project.\n\n"
    "Type hints were added in Python 3.5 to support static analysis.\n\n"
    "Async/await was introduced in Python 3.5 to express concurrent IO clearly.\n\n"
    "Numpy provides fast n-dimensional arrays for scientific computing.\n\n"
    "Pandas builds dataframes on top of numpy for tabular analysis.\n\n"
    "Rust is a systems language focused on memory safety without a garbage collector.\n\n"
    "Cargo is the official package manager and build tool for Rust projects."
)

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
        provider="openai", model=EMBED_MODEL, client=openai_client, batch_size=8
    )

    print(f"DEFAULT_DISTANCE_THRESHOLD = {DEFAULT_DISTANCE_THRESHOLD}")

    print("\n=== Test 1: percentile sweep on representative sample ===")
    for percentile in (0.5, 0.75, 0.85, 0.95, 1.0):
        t = await calibrate_distance_threshold(
            CALIBRATION_SAMPLE, embedder=embedder, percentile=percentile, margin=0.0
        )
        print(f"  p={percentile:.2f} margin=0.00 -> {t:.4f}")

    print("\n=== Test 2: default calibration (p=0.85, margin=0.05) ===")
    threshold = await calibrate_distance_threshold(CALIBRATION_SAMPLE, embedder=embedder)
    print(f"  calibrated threshold = {threshold:.4f}")
    assert 0.45 < threshold < 0.75, "expected threshold between intra and inter topic"

    print("\n=== Test 3: from_sample produces sane multi-topic grouping ===")
    chunker = await SemanticChunker.from_sample(CALIBRATION_SAMPLE, embedder=embedder)
    print(f"  auto threshold = {chunker.distance_threshold:.4f}")
    chunks = await chunker.chunk(DOCUMENT, page=1, source="calibration_test.md")
    print(f"  -> {len(chunks)} chunks")
    for c in chunks:
        n_paras = c.text.count("\n\n") + 1
        print(
            f"    idx={c.metadata.chunk_index} paragraphs={n_paras} "
            f"hash={c.content_hash[:10]}... :: {c.text[:60]!r}..."
        )
    assert len(chunks) == 3, f"Expected 3 topic groups (Python/Rome/Quantum), got {len(chunks)}"

    print("\n=== Test 4: multi-document sample (Sequence[str]) ===")
    multi = await calibrate_distance_threshold(
        [CALIBRATION_SAMPLE, DOCUMENT], embedder=embedder, percentile=0.5, margin=0.0
    )
    print(f"  median of all adjacent distances = {multi:.4f}")
    assert 0.0 <= multi <= 2.0

    print("\n=== Test 5: fallback when sample has no adjacent pairs ===")
    fallback = await calibrate_distance_threshold(
        "only one paragraph here.", embedder=embedder, fallback_threshold=0.42
    )
    assert fallback == 0.42
    print(f"  OK: fallback returned {fallback}")

    print("\n=== Test 6: validation errors ===")
    for kwargs, label in [
        ({"percentile": 1.5}, "percentile > 1.0"),
        ({"margin": -0.1}, "negative margin"),
        ({"fallback_threshold": 3.0}, "fallback out of range"),
    ]:
        try:
            await calibrate_distance_threshold(CALIBRATION_SAMPLE, embedder=embedder, **kwargs)
        except ValueError as exc:
            print(f"  OK: {label} -> ValueError: {exc}")
        else:
            raise AssertionError(f"Expected ValueError for {label}")

    print("\nALL CALIBRATION TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
