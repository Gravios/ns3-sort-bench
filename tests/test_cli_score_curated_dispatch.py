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


def test_find_raw_clu_prefers_untagged_canonical(tmp_path):
    """klusters_export writes `sim_session.clu.standard.1` with NO
    trailing stage tag; the curated .clu is the one with `.curated`
    tacked on.  `_find_raw_clu` must find the un-tagged canonical
    BEFORE falling through to any tag-suffixed scan — otherwise every
    GT and raw-sort lookup under a klusters_<sorter>/<item>/ directory
    comes up empty (the hc1_full run's symptom).
    """
    from ns3_sort_bench.cli.commands.score_curated_cmd import _find_raw_clu

    sess = tmp_path / "klusters_fiber_kit" / "item1"
    sess.mkdir(parents=True)
    # What klusters_export actually writes:
    untagged = sess / "sim_session.clu.standard.1"
    untagged.write_bytes(b"\x00\x00\x00\x00")   # content doesn't matter
    # The curated sibling the user created in Klusters:
    curated = sess / "sim_session.clu.standard.1.curated"
    curated.write_bytes(b"\x00\x00\x00\x00")

    got = _find_raw_clu(str(sess), "fiber_kit")
    assert got == str(untagged), (
        f"_find_raw_clu returned {got!r}; must prefer the un-tagged "
        f"canonical ({untagged}) over the curated or the empty scan."
    )


def test_find_raw_clu_falls_back_to_tagged_when_no_untagged(tmp_path):
    """When an older run or a third-party export wrote the raw .clu
    with the mode's end-of-chain tag (e.g. `.clu.standard.1.exp_final`)
    and NO un-tagged canonical sibling, `_find_raw_clu` must still pick
    it up via the documented-tag path or the glob fallback.
    """
    from ns3_sort_bench.cli.commands.score_curated_cmd import _find_raw_clu

    sess = tmp_path / "klusters_fiber_kit" / "item1"
    sess.mkdir(parents=True)
    tagged = sess / "sim_session.clu.standard.1.exp_final"
    tagged.write_bytes(b"\x00\x00\x00\x00")
    curated = sess / "sim_session.clu.standard.1.curated"
    curated.write_bytes(b"\x00\x00\x00\x00")

    got = _find_raw_clu(str(sess), "fiber_kit")
    assert got == str(tagged)


def test_find_raw_clu_does_not_return_the_curated_sibling(tmp_path):
    """Under no circumstance may `_find_raw_clu` return the curated
    .clu — scoring raw vs curated against GT would then compare the
    curated against itself and report perfect accuracy spuriously.
    """
    from ns3_sort_bench.cli.commands.score_curated_cmd import _find_raw_clu

    sess = tmp_path / "klusters_fiber_kit" / "item1"
    sess.mkdir(parents=True)
    curated = sess / "sim_session.clu.standard.1.curated"
    curated.write_bytes(b"\x00\x00\x00\x00")

    got = _find_raw_clu(str(sess), "fiber_kit")
    assert got is None, (
        "no raw .clu exists but _find_raw_clu returned "
        f"{got!r} — probably the curated sibling."
    )
