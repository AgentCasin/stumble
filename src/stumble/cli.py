import argparse

from .cleaner import load_documents
from .engine import SearchEngine


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stumble", description="Local document search"
    )
    parser.add_argument("query")
    parser.add_argument("--path", default=".", help="folder of .txt/.md documents")
    parser.add_argument(
        "--strategy", default="overlap", choices=["overlap", "bm25", "tfidf"]
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--show-snippets", action="store_true")
    args = parser.parse_args()
    engine = SearchEngine(load_documents(args.path), strategy=args.strategy)
    for r in engine.search(args.query, top_k=args.top_k):
        print(f"{r.score:6.3f}  {r.filename}")


if __name__ == "__main__":
    main()
