"""Smoke tests for `ns3sb score-curated` — mostly that `args.fn` is
wired, which the dispatcher in cli/main.py requires.

A past copy of score_curated_cmd.register() wrote ``set_defaults(func=run)``
while every other command uses ``fn=``, so invoking the command raised
AttributeError on args.fn before any work could happen.  This file pins
the fn attribute and runs the command on a nearly-empty directory to
confirm it returns a sensible exit code instead of crashing.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout, redirect_stderr

import pytest


def test_score_curated_subcommand_has_fn_attr():
    """The score-curated subparser's set_defaults must populate
    `args.fn`, not `args.func` — the dispatcher in cli/main.py reads
    `args.fn`.  A regression here would make the whole command crash
    with AttributeError on any invocation.
    """
    from ns3_sort_bench.cli import main as main_mod
    # Build the parser the same way main() does, then parse a minimal
    # invocation and check the resolved attribute.
    import argparse
    ap = argparse.ArgumentParser(prog="ns3sb")
    sub = ap.add_subparsers(dest="cmd")
    from ns3_sort_bench.cli.commands import score_curated_cmd
    score_curated_cmd.register(sub)
    args = ap.parse_args(["score-curated", "/tmp/does_not_matter"])
    assert hasattr(args, "fn"), \
        "score-curated register() must set_defaults(fn=...), not func=..."
    assert callable(args.fn)


def test_score_curated_cli_runs_without_crashing(tmp_path):
    """End-to-end: `ns3sb score-curated <empty_dir>` should NOT crash
    with AttributeError.  A good-case completion isn't the goal (no
    curated .clu exists, so there's nothing to score); we just want
    the dispatcher to reach the command function without blowing up.
    """
    from ns3_sort_bench.cli.main import main
    # The command only cares that output_dir is a real directory;
    # an empty one means it finds zero klusters_<sorter>/<item>/ dirs
    # and exits cleanly (or at worst non-zero without a stack trace).
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with redirect_stdout(buf_out), redirect_stderr(buf_err):
        rc = main(["score-curated", str(tmp_path)])
    # Any non-crashing return code is fine — we only catch the
    # historical AttributeError regression here.
    assert isinstance(rc, int)
    # And the error stream must NOT mention AttributeError.
    assert "AttributeError" not in buf_err.getvalue()
    assert "AttributeError" not in buf_out.getvalue()
