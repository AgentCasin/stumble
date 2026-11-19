"""stumble: fast, private, local document search for .txt and .md notes."""

from .engine import Result, SearchEngine  # public API

__all__ = ["Result", "SearchEngine"]
