"""`ns3sb inspect <corpus>` — load one corpus and print its shape."""
from __future__ import annotations

from .._util import eprint, resolve_corpus_by_key


def cmd_inspect(args) -> int:
    try:
        corpus = resolve_corpus_by_key(args.corpus)
    except KeyError as e:
        eprint(str(e))
        return 1
    except Exception as e:
        eprint(f"{type(e).__name__}: {e}")
        return 2
    rec = corpus.recording
    gt = corpus.ground_truth
    # SpikeInterface BaseRecording API; wrap each access in try to be
    # defensive against non-SI recordings (e.g. the Propper simulator).
    def _safe(obj, m, *a, **k):
        try:
            return getattr(obj, m)(*a, **k) if callable(getattr(obj, m, None)) else getattr(obj, m, None)
        except Exception:
            return None

    print(f"corpus      : {corpus.name}")
    print(f"kind        : {corpus.kind}")
    if corpus.description:
        print(f"description : {corpus.description}")
    nch = _safe(rec, "get_num_channels")
    fs = _safe(rec, "get_sampling_frequency")
    nsamp = _safe(rec, "get_num_samples")
    print(f"recording   : {type(rec).__name__}")
    if nch:    print(f"  channels  : {nch}")
    if fs:     print(f"  fs        : {fs} Hz")
    if nsamp:  print(f"  samples   : {nsamp:,} ({nsamp / fs:.1f} s)" if fs else f"  samples   : {nsamp:,}")

    print(f"ground_truth: {gt.n_units()} unit(s), {gt.n_spikes():,} spike(s)")
    for u in list(gt.unit_ids)[:10]:
        n = len(gt.spike_times[u])
        print(f"  unit {u}: {n:,} spikes")
    if gt.n_units() > 10:
        print(f"  ... ({gt.n_units() - 10} more)")

    if corpus.ns3_yaml:
        print("ns3_yaml    : attached (Contract validation enabled)")
    if corpus.extras:
        print(f"extras      : {sorted(corpus.extras)}")
    return 0


def register(subparsers) -> None:
    ap = subparsers.add_parser("inspect", help="load one corpus and summarise it")
    ap.add_argument("corpus", help="e.g. paired_crcns_hc1:d533101")
    ap.set_defaults(fn=cmd_inspect)
