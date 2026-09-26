"""
Sentence-boundary-aware chunker using NLTK.
"""

from __future__ import annotations

from app.chunking.base import BaseChunker
from app.domain.chunks.models import Chunk
from app.domain.documents.models import Document


class SentenceChunker(BaseChunker):
    """
    Group complete sentences into chunks without splitting mid-sentence.

    Parameters
    ----------
    sentences_per_chunk:
        Target number of sentences per chunk.
    sentence_overlap:
        Number of sentences carried over into the next chunk.
    language:
        NLTK sentence tokenizer language.
    """

    def __init__(
        self,
        sentences_per_chunk: int = 5,
        sentence_overlap: int = 1,
        language: str = "english",
    ) -> None:
        self.sentences_per_chunk = sentences_per_chunk
        self.sentence_overlap = sentence_overlap
        self.language = language

    def _tokenize(self, text: str) -> list[str]:
        try:
            import nltk  # type: ignore[import-untyped]
            try:
                sentences = nltk.sent_tokenize(text, language=self.language)
            except LookupError:
                nltk.download("punkt_tab", quiet=True)
                nltk.download("punkt", quiet=True)
                sentences = nltk.sent_tokenize(text, language=self.language)
            return sentences
        except ImportError as exc:
            raise ImportError("nltk is required. Run: pip install nltk") from exc

    def split(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text.strip():
            return []

        sentences = self._tokenize(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        idx = 0
        start = 0

        while start < len(sentences):
            end = min(start + self.sentences_per_chunk, len(sentences))
            chunk_sentences = sentences[start:end]
            chunk_text = " ".join(chunk_sentences).strip()

            if chunk_text:
                chunks.append(
                    self._make_chunk(
                        document,
                        content=chunk_text,
                        index=idx,
                        chunker_type="sentence",
                    )
                )
                idx += 1

            step = max(self.sentences_per_chunk - self.sentence_overlap, 1)
            start += step

        return chunks
