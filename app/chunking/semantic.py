"""
Semantic chunker — groups sentences by embedding similarity.

Uses a lightweight local sentence-transformers model to embed each
sentence, then starts a new chunk whenever consecutive sentences cross
a cosine-similarity threshold.  Produces topic-coherent chunks at the
cost of an embedding call during ingestion.
"""

from __future__ import annotations

import math

from app.chunking.base import BaseChunker
from app.domain.chunks.models import Chunk
from app.domain.documents.models import Document


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    ma = math.sqrt(sum(x * x for x in a))
    mb = math.sqrt(sum(x * x for x in b))
    if ma == 0 or mb == 0:
        return 0.0
    return dot / (ma * mb)


class SemanticChunker(BaseChunker):
    """
    Groups sentences by semantic similarity using embeddings.

    Parameters
    ----------
    embedding_model:
        sentence-transformers model name or local path.
    similarity_threshold:
        Cosine similarity below which a new chunk is started.
    max_chunk_size:
        Hard upper limit on chunk size in characters.
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.5,
        max_chunk_size: int = 2000,
    ) -> None:
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        self.max_chunk_size = max_chunk_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for semantic chunking. "
                    "Run: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.embedding_model)
        return self._model

    def _tokenize_sentences(self, text: str) -> list[str]:
        try:
            import nltk  # type: ignore[import-untyped]
            try:
                return nltk.sent_tokenize(text)
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                nltk.download("punkt", quiet=True)
                return nltk.sent_tokenize(text)
        except ImportError:
            # Basic fallback: split on ". "
            return [s.strip() for s in text.split(". ") if s.strip()]

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sentences = self._tokenize_sentences(text)
        if not sentences:
            return []

        model = self._get_model()
        embeddings: list[list[float]] = model.encode(sentences, show_progress_bar=False).tolist()

        # Group sentences into chunks
        groups: list[list[str]] = [[sentences[0]]]

        for i in range(1, len(sentences)):
            sim = _cosine(embeddings[i - 1], embeddings[i])
            current_group_text = " ".join(groups[-1])

            if sim < self.similarity_threshold or len(current_group_text) >= self.max_chunk_size:
                groups.append([sentences[i]])
            else:
                groups[-1].append(sentences[i])

        return [
            self._make_chunk(
                document,
                content=" ".join(group).strip(),
                index=i,
                chunker_type="semantic",
            )
            for i, group in enumerate(groups)
            if " ".join(group).strip()
        ]
