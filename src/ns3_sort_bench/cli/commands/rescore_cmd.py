"""``ns3sb rescore`` — score a run's saved sorts against the reference as it
is now.

The reference of a lab corpus is a sort under curation: every merge or
split changes the ``.clu`` the corpus key names, and a run's scores
describe the reference as it was when the run started.  ``ns3sb run``
saves every sort to ``<run>/sorts/<item>/<adapter>.npz`` and the corpus
key of every item to ``<run>/corpora.json``; this command resolves each
key again (re-reading the reference from disk), scores every saved sort
with the scorers, and rewrites the item's rows in ``scores.csv``.  No
sorter runs.  The previous ``scores.csv`` is kept as
``scores.prev.csv`` and ``rescore.json`` records when, against which
key, and how big the reference was.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil

from .._util import eprint, import_registries, split_csv


def _scorers_in(scores_csv: str):
    import pandas as pd
    try:
        df = pd.read_csv(scores_csv, keep_default_na=False)
        return sorted(set(df["scorer"])) if "scorer" in df else []
    except (OSError, ValueError):
        return []


def cmd_rescore(args) -> int:
    import_registries()
    import pandas as pd
    from ... import scorer_registry
    from ...runners.sorter_grid import _resolve_corpus
    from ...sort_store import read_corpora, record_corpora, saved_sorts

    run_dir = args.run_dir
    scores_csv = os.path.join(run_dir, "scores.csv")

    keys = read_corpora(run_dir)
    if args.corpus:
        given = split_csv(args.corpus)
        keys = {}
        for key in given:
            try:
                c = _resolve_corpus(key)
            except KeyError as e:
                eprint(str(e))
                return 1
            keys[c.name] = key
    if not keys:
        eprint(f"no corpus keys for {run_dir}: it has no corpora.json (runs before "
               "the rescore command).  Pass the run's --corpus key(s).")
        return 1

    scorer_names = (split_csv(args.scorers) or _scorers_in(scores_csv)
                    or [n for n in ("pairwise", "per_cell") if n in scorer_registry])
    missing = [n for n in scorer_names if n not in scorer_registry]
    if missing:
        eprint(f"unknown scorer(s): {missing}")
        return 1
    scorers = [scorer_registry[n] for n in scorer_names]

    new_rows, provenance, rescored_items = [], [], []
    for item, key in sorted(keys.items()):
        sorts = saved_sorts(run_dir, item)
        if not sorts:
            eprint(f"  {item}: no saved sorts under {run_dir}/sorts/ "
                   "(runs before saved sorts need one more `ns3sb run`)")
            continue
        corpus = _resolve_corpus(key)
        ref = corpus.ground_truth
        if ref is None:
            eprint(f"  {item}: {key} has no reference to score against")
            continue
        print(f"[rescore] {item}: reference {ref.n_units()} units, "
              f"{ref.n_spikes():,} spikes; {len(sorts)} saved sorts")
        for cand in sorts:
            for sc in scorers:
                try:
                    new_rows.extend(sc(ref, cand).to_rows())
                except Exception as e:
                    eprint(f"    {cand.adapter} / {sc.name}: "
                           f"{type(e).__name__}: {e}")
        rescored_items.append(item)
        provenance.append(dict(corpus_item=item, corpus_key=key,
                               ref_units=int(ref.n_units()),
                               ref_spikes=int(ref.n_spikes()),
                               sorts=[c.adapter for c in sorts]))
        record_corpora(run_dir, {item: key})

    if not rescored_items:
        eprint("nothing rescored")
        return 1

    kept = []
    if os.path.exists(scores_csv) and os.path.getsize(scores_csv) > 1:
        shutil.copy2(scores_csv, os.path.join(run_dir, "scores.prev.csv"))
        old = pd.read_csv(scores_csv, keep_default_na=False)
        kept = old[~old["corpus_item"].astype(str).isin(rescored_items)].to_dict("records")
    df = pd.DataFrame(kept + new_rows)
    df.to_csv(scores_csv, index=False)

    with open(os.path.join(run_dir, "rescore.json"), "w") as f:
        json.dump(dict(when=_dt.datetime.now().isoformat(timespec="seconds"),
                       scorers=scorer_names, items=provenance), f, indent=2)
    print(f"wrote {scores_csv} ({len(new_rows)} new rows; the previous one is "
          f"scores.prev.csv)\n  ns3sb summary {run_dir}")
    return 0


def register(subparsers) -> None:
    ap = subparsers.add_parser(
        "rescore",
        help="score a run's saved sorts against the reference as it is now "
             "(no sorting)")
    ap.add_argument("run_dir", help="a run's --out folder (holds sorts/ and scores.csv)")
    ap.add_argument("--corpus", default="",
                    help="corpus key(s) to resolve the reference from (default: the "
                         "run's corpora.json; needed for runs made before it)")
    ap.add_argument("--scorers", default="",
                    help="comma-separated scorers (default: the ones in scores.csv)")
    ap.set_defaults(fn=cmd_rescore)
