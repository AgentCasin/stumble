"""Document loading and lightweight text normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True)
class Document:
    """A document identified by its filename and containing its text."""

    filename: str
    text: str


def load_documents(path: str | Path) -> list[Document]:
    """Load .txt and .md documents recursively from a directory."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Corpus directory does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Corpus path is not a directory: {root}")

    files = sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() in {".txt", ".md"}
        ),
        key=lambda candidate: (
            candidate.name,
            candidate.relative_to(root).as_posix(),
        ),
    )
    if not files:
        raise ValueError(f"No .txt or .md documents found in corpus directory: {root}")

    return [
        Document(filename=file_path.name, text=file_path.read_text(encoding="utf-8"))
        for file_path in files
    ]


def tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric and apostrophe-delimited tokens."""
    return _TOKEN_PATTERN.findall(text.lower())


def stem(token: str) -> str:
    """Apply a small suffix-stripping stemmer to a single token."""
    word = token.lower()
    if len(word) > 4 and word.endswith("ying"):
        return word[:-4] + "y"

    if len(word) > 4 and word.endswith("ing"):
        stemmed = word[:-3]
        return _remove_doubled_consonant(stemmed)

    if len(word) > 3 and word.endswith("ed"):
        stemmed = word[:-2]
        return _remove_doubled_consonant(stemmed)

    if len(word) > 4 and word.endswith("es"):
        return word[:-2]

    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]

    return word


def clean(text: str) -> list[str]:
    """Tokenize text, remove stopwords, and stem the remaining tokens."""
    return [stem(token) for token in tokenize(text) if token not in STOPWORDS]


def _remove_doubled_consonant(word: str) -> str:
    """Remove a doubled final consonant after an inflectional suffix."""
    if len(word) > 2 and word[-1] == word[-2] and word[-1] not in "aeiou":
        return word[:-1]
    return word
