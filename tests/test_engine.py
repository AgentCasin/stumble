import json
from pathlib import Path

import pytest

from stumble.cleaner import Document
from stumble.engine import Result, SearchEngine


@pytest.fixture
def corpus() -> list[Document]:
    return [
        Document("alpha.txt", "apple banana apple"),
        Document("beta.txt", "banana cherry"),
        Document("gamma.txt", "cherry date"),
    ]


def test_result_holds_filename_and_score() -> None:
    result = Result("notes.md", 0.5)

    assert result.filename == "notes.md"
    assert result.score == 0.5


def test_engine_rejects_unknown_strategy(corpus: list[Document]) -> None:
    with pytest.raises(ValueError, match="strategy"):
        SearchEngine(corpus, strategy="unknown")


def test_engine_rejects_empty_corpus() -> None:
    with pytest.raises(ValueError, match="corpus"):
        SearchEngine([])


def test_overlap_returns_documents_with_query_terms(corpus: list[Document]) -> None:
    results = SearchEngine(corpus).search("apple cherry")

    assert [result.filename for result in results] == [
        "alpha.txt",
        "beta.txt",
        "gamma.txt",
    ]
    assert all(0.0 <= result.score <= 1.0 for result in results)


def test_overlap_scores_distinct_query_terms(corpus: list[Document]) -> None:
    results = SearchEngine(corpus).search("apple apple banana")

    assert [(result.filename, result.score) for result in results] == [
        ("alpha.txt", 1.0),
        ("beta.txt", 0.5),
    ]


def test_search_returns_empty_for_stopwords_only_query(corpus: list[Document]) -> None:
    assert SearchEngine(corpus).search("the and is") == []


def test_search_returns_empty_when_no_terms_match(corpus: list[Document]) -> None:
    assert SearchEngine(corpus).search("kiwi") == []


def test_search_limits_results_to_top_k(corpus: list[Document]) -> None:
    assert len(SearchEngine(corpus).search("banana cherry", top_k=1)) == 1


def test_search_orders_equal_scores_by_filename() -> None:
    documents = [
        Document("zeta.txt", "unique"),
        Document("alpha.txt", "unique"),
    ]

    results = SearchEngine(documents).search("unique")

    assert [result.filename for result in results] == ["alpha.txt", "zeta.txt"]


@pytest.mark.parametrize("strategy", ["overlap", "bm25", "tfidf"])
def test_all_strategies_return_normalized_scores(
    corpus: list[Document], strategy: str
) -> None:
    results = SearchEngine(corpus, strategy=strategy).search("apple banana cherry")

    assert results
    assert all(0.0 <= result.score <= 1.0 for result in results)
    assert results[0].score == 1.0


def test_bm25_prefers_repeated_term_when_document_lengths_equal() -> None:
    documents = [
        Document("one.txt", "apple apple banana"),
        Document("two.txt", "apple banana banana"),
    ]

    results = SearchEngine(documents, strategy="bm25").search("apple")

    assert results[0].filename == "one.txt"


def test_tfidf_prefers_the_more_specific_query_match() -> None:
    documents = [
        Document("common.txt", "apple banana"),
        Document("specific.txt", "apple"),
    ]

    results = SearchEngine(documents, strategy="tfidf").search("apple banana")

    assert results[0].filename == "common.txt"


def test_document_synonyms_expand_indexed_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    (data_directory / "synonyms.json").write_text(
        json.dumps({"automobile": ["car"]}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    results = SearchEngine([Document("vehicle.txt", "car")]).search("automobile")

    assert [result.filename for result in results] == ["vehicle.txt"]


def test_missing_synonyms_file_does_not_prevent_search(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    results = SearchEngine([Document("vehicle.txt", "car")]).search("car")

    assert [result.filename for result in results] == ["vehicle.txt"]


def test_query_synonyms_expand_query_terms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    (data_directory / "synonyms.json").write_text(
        json.dumps({"car": ["automobile"]}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    results = SearchEngine([Document("vehicle.txt", "automobile")]).search("car")

    assert [result.filename for result in results] == ["vehicle.txt"]


def test_synonym_expansion_deduplicates_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    (data_directory / "synonyms.json").write_text(
        json.dumps({"car": ["automobile", "automobile"]}), encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)

    results = SearchEngine([Document("vehicle.txt", "car")]).search("car car")

    assert len(results) == 1
    assert results[0].score == 1.0
