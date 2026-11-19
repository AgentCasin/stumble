from pathlib import Path

import pytest

from stumble.cleaner import Document, clean, load_documents, stem, tokenize


def test_document_holds_filename_and_text() -> None:
    document = Document("notes.md", "A note")

    assert document.filename == "notes.md"
    assert document.text == "A note"


def test_load_documents_reads_text_and_markdown_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "one.txt").write_text("first", encoding="utf-8")
    (nested / "two.md").write_text("second", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("ignored", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [(document.filename, document.text) for document in documents] == [
        ("one.txt", "first"),
        ("two.md", "second"),
    ]


def test_load_documents_uses_deterministic_filename_order(tmp_path: Path) -> None:
    (tmp_path / "z.md").write_text("z", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.filename for document in documents] == ["a.txt", "z.md"]


def test_load_documents_rejects_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No .txt or .md documents"):
        load_documents(tmp_path)


def test_load_documents_rejects_non_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("text", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_documents(file_path)


def test_tokenize_lowercases_and_matches_allowed_characters() -> None:
    assert tokenize("Don't stop, 42 times! Café") == [
        "don't",
        "stop",
        "42",
        "times",
        "caf",
    ]


def test_tokenize_returns_empty_for_punctuation() -> None:
    assert tokenize("... -- !!!") == []


def test_stopwords_are_removed_by_clean() -> None:
    assert clean("The cat is on the mat") == ["cat", "mat"]


def test_stem_removes_plural_s() -> None:
    assert stem("cats") == "cat"


def test_stem_removes_es() -> None:
    assert stem("boxes") == "box"


def test_stem_converts_ying_to_y() -> None:
    assert stem("flying") == "fly"


def test_stem_removes_ing_and_reduces_doubled_consonant() -> None:
    assert stem("running") == "run"


def test_stem_removes_ed_and_reduces_doubled_consonant() -> None:
    assert stem("stopped") == "stop"


def test_stem_keeps_single_consonant_before_ing() -> None:
    assert stem("walking") == "walk"


def test_stem_preserves_short_words() -> None:
    assert stem("is") == "is"


def test_stem_handles_case_insensitively() -> None:
    assert stem("RUNNING") == "run"


def test_clean_tokenizes_filters_and_stems() -> None:
    assert clean("The runners were running and stopped.") == [
        "runner",
        "run",
        "stop",
    ]
