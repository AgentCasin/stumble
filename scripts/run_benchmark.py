"""Evaluate document search against a labeled question set."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from stumble.cleaner import load_documents
from stumble.engine import SearchEngine

_STRATA = ("verbatim", "paraphrase", "colloquial", "impossible")
_ALL_QUESTIONS_LABEL = "All questions"


@dataclass(frozen=True)
class BenchmarkQuestion:
    """A query and the filenames considered relevant to it."""

    identifier: int
    query: str
    stratum: str
    expected_file: str
    also_acceptable: tuple[str, ...]


@dataclass(frozen=True)
class MetricSummary:
    """Aggregate ranking metrics for a set of benchmark questions."""

    accuracy_at_1: float
    accuracy_at_3: float
    mrr: float
    recall_at_3: float


class _SearchResult(Protocol):
    """The result fields required by the benchmark evaluator."""

    @property
    def filename(self) -> str:
        """Return the result filename."""
        ...


class _SearchEngine(Protocol):
    """The search interface required by the benchmark evaluator."""

    def search(self, query: str, top_k: int = 3) -> Sequence[_SearchResult]:
        """Return ranked results for a query."""
        ...


def load_questions(path: str | Path) -> list[BenchmarkQuestion]:
    """Load and validate benchmark questions from a JSON file."""
    questions, _ = _read_question_file(Path(path))
    return questions


def calculate_metrics(
    questions: Sequence[BenchmarkQuestion],
    engine: _SearchEngine,
    top_k: int = 3,
) -> dict[str, MetricSummary]:
    """Calculate overall and per-stratum accuracy, MRR, and recall."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    scored_questions = [
        (question, _calculate_question_metrics(question, engine, top_k))
        for question in questions
    ]
    summaries = {
        _ALL_QUESTIONS_LABEL: _average_metrics(
            [metrics for _, metrics in scored_questions]
        )
    }
    for stratum in _STRATA:
        summaries[stratum] = _average_metrics(
            [
                metrics
                for question, metrics in scored_questions
                if question.stratum == stratum
            ]
        )
    return summaries


def render_report(summaries: dict[str, MetricSummary]) -> str:
    """Render benchmark metric summaries as a Markdown table."""
    labels = (_ALL_QUESTIONS_LABEL, *_STRATA)
    lines = [
        "| Stratum | accuracy@1 | accuracy@3 | MRR | recall@3 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in labels:
        summary = summaries[label]
        lines.append(
            f"| {label} | {summary.accuracy_at_1:.3f} | "
            f"{summary.accuracy_at_3:.3f} | {summary.mrr:.3f} | "
            f"{summary.recall_at_3:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI and write results.md."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        help="document directory (defaults to the corpus named in the questions file)",
    )
    parser.add_argument(
        "--questions",
        default="data/test_questions.json",
        help="benchmark questions JSON file",
    )
    parser.add_argument(
        "--strategy",
        choices=("overlap", "bm25", "tfidf"),
        default="overlap",
        help="ranking strategy",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="minimum number of ranked results to evaluate per question",
    )
    arguments = parser.parse_args(argv)
    if arguments.top_k < 1:
        parser.error("--top-k must be at least 1")

    question_path = Path(arguments.questions)
    try:
        questions, corpus_name = _read_question_file(question_path)
        corpus_path = _resolve_corpus_path(arguments.corpus, corpus_name, question_path)
        documents = load_documents(corpus_path)
        engine = SearchEngine(documents, strategy=arguments.strategy)
        summaries = calculate_metrics(questions, engine, arguments.top_k)
        report = render_report(summaries)
        Path("results.md").write_text(report, encoding="utf-8")
    except (OSError, TypeError, ValueError) as error:
        parser.error(str(error))

    print(report, end="")
    return 0


def _read_question_file(
    path: Path,
) -> tuple[list[BenchmarkQuestion], str | None]:
    """Read question entries and the optional corpus path from JSON."""
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read questions file {path}: {error}") from error

    if not isinstance(data, dict):
        raise TypeError(f"Questions file must contain a JSON object: {path}")
    raw_questions = data.get("questions")
    if not isinstance(raw_questions, list):
        raise TypeError(f"Questions file must contain a questions list: {path}")

    corpus_name = data.get("corpus")
    if corpus_name is not None and not isinstance(corpus_name, str):
        raise TypeError(f"Questions file corpus must be a string: {path}")

    questions: list[BenchmarkQuestion] = []
    for position, raw_question in enumerate(raw_questions, start=1):
        if not isinstance(raw_question, dict):
            raise TypeError(f"Question {position} must be a JSON object")
        identifier = raw_question.get("id", position)
        query = raw_question.get("query")
        stratum = raw_question.get("stratum")
        expected_file = raw_question.get("expected_file")
        acceptable = raw_question.get("also_acceptable", [])
        if not isinstance(identifier, int) or isinstance(identifier, bool):
            raise TypeError(f"Question {position} id must be an integer")
        if not isinstance(query, str) or not query:
            raise TypeError(f"Question {position} query must be a non-empty string")
        if not isinstance(stratum, str) or stratum not in _STRATA:
            raise ValueError(
                f"Question {position} stratum must be one of {', '.join(_STRATA)}"
            )
        if not isinstance(expected_file, str) or not expected_file:
            raise TypeError(
                f"Question {position} expected_file must be a non-empty string"
            )
        if not isinstance(acceptable, list) or not all(
            isinstance(filename, str) and filename for filename in acceptable
        ):
            raise TypeError(
                f"Question {position} also_acceptable must be a list of filenames"
            )
        questions.append(
            BenchmarkQuestion(
                identifier=identifier,
                query=query,
                stratum=stratum,
                expected_file=expected_file,
                also_acceptable=tuple(acceptable),
            )
        )
    return questions, corpus_name


def _resolve_corpus_path(
    override: str | None, corpus_name: str | None, questions_path: Path
) -> Path:
    """Resolve the corpus override or path named in the questions file."""
    if override is not None:
        return Path(override)
    if corpus_name is None:
        raise ValueError(
            "Specify --corpus or include a corpus path in the questions file"
        )

    corpus_path = Path(corpus_name)
    if corpus_path.exists():
        return corpus_path
    relative_to_questions = questions_path.parent / corpus_path
    if relative_to_questions.exists():
        return relative_to_questions
    raise FileNotFoundError(
        f"Corpus directory does not exist: {corpus_path} "
        f"(also checked {relative_to_questions})"
    )


def _calculate_question_metrics(
    question: BenchmarkQuestion, engine: _SearchEngine, top_k: int
) -> MetricSummary:
    """Compute the four benchmark metrics for one question."""
    requested_results = max(3, top_k)
    relevant = {question.expected_file, *question.also_acceptable}
    results = engine.search(question.query, top_k=requested_results)
    filenames = [result.filename for result in results]
    top_three = filenames[:3]
    reciprocal_rank = 0.0
    for rank, filename in enumerate(filenames, start=1):
        if filename in relevant:
            reciprocal_rank = 1.0 / rank
            break
    return MetricSummary(
        accuracy_at_1=float(bool(filenames and filenames[0] in relevant)),
        accuracy_at_3=float(bool(relevant.intersection(top_three))),
        mrr=reciprocal_rank,
        recall_at_3=len(relevant.intersection(top_three)) / len(relevant),
    )


def _average_metrics(metrics: Sequence[MetricSummary]) -> MetricSummary:
    """Average per-question metric values, returning zero for an empty group."""
    if not metrics:
        return MetricSummary(0.0, 0.0, 0.0, 0.0)

    question_count = len(metrics)
    return MetricSummary(
        accuracy_at_1=sum(item.accuracy_at_1 for item in metrics) / question_count,
        accuracy_at_3=sum(item.accuracy_at_3 for item in metrics) / question_count,
        mrr=sum(item.mrr for item in metrics) / question_count,
        recall_at_3=sum(item.recall_at_3 for item in metrics) / question_count,
    )


if __name__ == "__main__":
    raise SystemExit(main())
