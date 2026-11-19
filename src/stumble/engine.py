"""In-memory ranked search over a collection of documents."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from stumble.cleaner import Document, clean

_STRATEGIES = frozenset({"overlap", "bm25", "tfidf"})
_BM25_K1 = 1.5
_BM25_B = 0.75


@dataclass(frozen=True)
class Result:
    """A search result containing a filename and normalized relevance score."""

    filename: str
    score: float


class SearchEngine:
    """Build an inverted index and rank documents for text queries."""

    def __init__(self, corpus: Iterable[Document], strategy: str = "overlap") -> None:
        """Initialize the index for a non-empty corpus and ranking strategy."""
        if strategy not in _STRATEGIES:
            choices = ", ".join(sorted(_STRATEGIES))
            raise ValueError(f"Unknown search strategy {strategy!r}; choose {choices}")

        documents = tuple(corpus)
        if not documents:
            raise ValueError("Search corpus must contain at least one document")

        for document in documents:
            if not isinstance(document, Document):
                raise TypeError("Search corpus entries must be Document instances")

        self._strategy = strategy
        self._synonyms = self._load_synonyms(Path.cwd() / "data" / "synonyms.json")
        self._index: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_lengths: dict[str, int] = {}

        for document in documents:
            terms = self._expand_terms(clean(document.text))
            frequencies = Counter(terms)
            self._doc_lengths[document.filename] = len(terms)
            for term, frequency in frequencies.items():
                self._index[term][document.filename] = frequency

        self._document_count = len(documents)
        self._average_length = sum(self._doc_lengths.values()) / self._document_count

    def search(self, query: str, top_k: int = 3) -> list[Result]:
        """Return up to top_k matching documents ranked by relevance."""
        if top_k < 0:
            raise ValueError("top_k must be zero or greater")
        if top_k == 0:
            return []

        query_terms = set(self._expand_terms(clean(query)))
        if not query_terms:
            return []

        candidate_names: set[str] = set()
        for term in query_terms:
            candidate_names.update(self._index.get(term, {}))
        if not candidate_names:
            return []

        raw_scores = {
            filename: self._score(filename, query_terms) for filename in candidate_names
        }
        maximum = max(raw_scores.values())
        if maximum <= 0.0:
            return []

        results = [
            Result(filename, score / maximum) for filename, score in raw_scores.items()
        ]
        results.sort(key=lambda result: (-result.score, result.filename))
        return results[:top_k]

    def _score(self, filename: str, query_terms: set[str]) -> float:
        """Calculate a document's raw score for a set of query terms."""
        if self._strategy == "overlap":
            matches = sum(filename in self._index.get(term, {}) for term in query_terms)
            return matches / len(query_terms)

        score = 0.0
        document_length = self._doc_lengths[filename]
        for term in query_terms:
            term_frequencies = self._index.get(term)
            if term_frequencies is None or filename not in term_frequencies:
                continue

            frequency = term_frequencies[filename]
            document_frequency = len(term_frequencies)
            if self._strategy == "tfidf":
                inverse_document_frequency = 1.0 + math.log(
                    self._document_count / document_frequency
                )
                score += frequency * inverse_document_frequency
            else:
                inverse_document_frequency = math.log(
                    1.0
                    + (self._document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                normalization = _BM25_K1 * (
                    1.0 - _BM25_B + _BM25_B * document_length / self._average_length
                )
                score += inverse_document_frequency * (
                    frequency * (_BM25_K1 + 1.0) / (frequency + normalization)
                )
        return score

    def _expand_terms(self, terms: Iterable[str]) -> list[str]:
        """Expand terms through the configured synonym relation."""
        expanded: list[str] = []
        for term in terms:
            expanded.append(term)
            expanded.extend(sorted(self._synonyms.get(term, ())))
        return expanded

    @staticmethod
    def _load_synonyms(path: Path) -> dict[str, frozenset[str]]:
        """Read and validate an optional JSON synonym mapping."""
        if not path.exists():
            return {}

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read synonyms file {path}: {error}") from error

        if not isinstance(data, dict):
            raise TypeError(f"Synonyms file must contain a JSON object: {path}")

        synonyms: dict[str, frozenset[str]] = {}
        for term, values in data.items():
            if (
                not isinstance(term, str)
                or not isinstance(values, list)
                or not all(isinstance(value, str) for value in values)
            ):
                raise ValueError(
                    f"Synonyms file entries must map strings to string lists: {path}"
                )

            normalized_term = clean(term)
            if len(normalized_term) != 1:
                raise ValueError(
                    f"Synonym keys must normalize to exactly one term: {term!r}"
                )
            normalized_values = {
                normalized for value in values for normalized in clean(value)
            }
            synonyms[normalized_term[0]] = frozenset(normalized_values)
        return synonyms
