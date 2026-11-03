"""Shared helpers for the ns3sb CLI commands."""
from __future__ import annotations

import sys
from typing import List, Optional


def eprint(*args, **kwargs) -> None:
    """Print to stderr."""
    print(*args, file=sys.stderr, **kwargs)


def split_csv(s: Optional[str]) -> List[str]:
    """Split a comma-separated argument into a cleaned list; '' → []."""
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def import_registries() -> None:
    """Trigger the self-registering adapter + scorer modules."""
    import ns3_sort_bench.adapters   # noqa: F401
    import ns3_sort_bench.scorers    # noqa: F401


def table(rows, headers=None, pad: int = 2) -> str:
    """Format rows as an aligned text table."""
    rows = [list(r) for r in rows]
    if headers:
        rows = [list(headers)] + rows
    if not rows:
        return ""
    cols = len(rows[0])
    widths = [max(len(str(r[i])) for r in rows) for i in range(cols)]
    sep = " " * pad
    out = [sep.join(str(r[i]).ljust(widths[i]) for i in range(cols)) for r in rows]
    if headers:
        out.insert(1, sep.join("-" * widths[i] for i in range(cols)))
    return "\n".join(out)


def resolve_corpus_by_key(key: str):
    """Pass through to runners.sorter_grid._resolve_corpus."""
    from ..runners.sorter_grid import _resolve_corpus
    return _resolve_corpus(key)
