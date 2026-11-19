import json
from pathlib import Path

from scripts.run_benchmark import (
    BenchmarkQuestion,
    MetricSummary,
    calculate_metrics,
    load_questions,
    render_report,
)
from stumble.engine import Result


class FakeEngine:
    """Return a fixed ranking without loading a document corpus."""

    def __init__(self, filenames: list[str]) -> None:
        """Store the ranked filenames."""
        self._filenames = filenames

    def search(self, query: str, top_k: int = 3) -> list[Result]:
        """Return the configured filenames in ranking order."""
        return [Result(filename, 1.0) for filename in self._filenames[:top_k]]


def _load_question(
    tmp_path: Path, also_acceptable: list[str] | None = None
) -> list[BenchmarkQuestion]:
    """Write and load one benchmark question in the temporary directory."""
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": 1,
                        "query": "find the route",
                        "stratum": "paraphrase",
                        "expected_file": "correct.md",
                        "also_acceptable": also_acceptable or [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return load_questions(questions_path)


def test_all_correct_engine_has_perfect_metrics(tmp_path: Path) -> None:
    """Return exact perfect metrics for a fully correct fake ranking."""
    questions = _load_question(tmp_path)

    summaries = calculate_metrics(questions, FakeEngine(["correct.md", "other.md"]))

    assert summaries["All questions"] == MetricSummary(1.0, 1.0, 1.0, 1.0)
    assert summaries["paraphrase"] == MetricSummary(1.0, 1.0, 1.0, 1.0)


def test_second_ranked_correct_file_has_half_mrr(tmp_path: Path) -> None:
    """Score a second-ranked relevant result with reciprocal rank one half."""
    questions = _load_question(tmp_path)

    summaries = calculate_metrics(questions, FakeEngine(["wrong.md", "correct.md"]))

    metrics = summaries["All questions"]
    assert metrics.accuracy_at_1 == 0.0
    assert metrics.accuracy_at_3 == 1.0
    assert metrics.mrr == 0.5
    assert metrics.recall_at_3 == 1.0


def test_unretrieved_correct_file_has_zero_mrr(tmp_path: Path) -> None:
    """Return zero MRR when no ranked result is relevant."""
    questions = _load_question(tmp_path)

    summaries = calculate_metrics(questions, FakeEngine(["wrong.md"]))

    assert summaries["All questions"].mrr == 0.0


def test_recall_at_three_counts_distinct_relevant_files(tmp_path: Path) -> None:
    """Average recall@3 across questions with multiple relevant filenames."""
    questions = _load_question(tmp_path, ["acceptable.md"])

    summaries = calculate_metrics(questions, FakeEngine(["acceptable.md"]))

    metrics = summaries["All questions"]
    assert metrics.accuracy_at_1 == 1.0
    assert metrics.mrr == 1.0
    assert metrics.recall_at_3 == 0.5


def test_report_contains_overall_and_each_stratum(tmp_path: Path) -> None:
    """Render the required metric columns for overall and per-stratum results."""
    questions = _load_question(tmp_path)
    summaries = calculate_metrics(questions, FakeEngine(["correct.md"]))

    report = render_report(summaries)

    assert "| Stratum | accuracy@1 | accuracy@3 | MRR | recall@3 |" in report
    assert "| All questions | 1.000 | 1.000 | 1.000 | 1.000 |" in report
    for stratum in ("verbatim", "paraphrase", "colloquial", "impossible"):
        assert f"| {stratum} |" in report


def test_load_questions_accepts_utf8_bom(tmp_path: Path) -> None:
    """Read question JSON saved with a Windows UTF-8 byte-order mark."""
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": 1,
                        "query": "find route",
                        "stratum": "verbatim",
                        "expected_file": "route.md",
                        "also_acceptable": [],
                    }
                ]
            }
        ),
        encoding="utf-8-sig",
    )

    assert load_questions(questions_path)[0].expected_file == "route.md"
