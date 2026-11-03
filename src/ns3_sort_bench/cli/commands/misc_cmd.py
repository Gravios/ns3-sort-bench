"""Thin pass-throughs: `ns3sb lattice`, `ns3sb render`, `ns3sb doctor`."""
from __future__ import annotations

import sys
from pathlib import Path

from .._util import eprint, import_registries


# ─── lattice ─────────────────────────────────────────────────────────────────
def cmd_lattice(args) -> int:
    """Delegate to runners.lattice (ablation sweep over the 32-config lattice)."""
    from ...runners import lattice
    sys.argv = ["ns3sb lattice", *args.rest]
    return lattice.main() or 0


def register_lattice(subparsers) -> None:
    ap = subparsers.add_parser("lattice",
                               help="run the fiber-kit lattice ablation sweep")
    ap.add_argument("rest", nargs="*",
                    help="forwarded to runners.lattice (see its --help)")
    ap.set_defaults(fn=cmd_lattice)


# ─── render ──────────────────────────────────────────────────────────────────
def cmd_render(args) -> int:
    """Render a run's scores.csv into an HTML dashboard."""
    from ...dashboard import render as render_mod
    run_dir = Path(args.run_dir)
    if not (run_dir / "scores.csv").exists():
        eprint(f"no scores.csv in {run_dir} — nothing to render")
        return 1
    sys.argv = ["ns3sb render", str(run_dir)]
    return render_mod.main() or 0


def register_render(subparsers) -> None:
    ap = subparsers.add_parser("render",
                               help="render HTML dashboard for a run directory")
    ap.add_argument("run_dir", help="directory holding scores.csv + sort_summaries.json")
    ap.set_defaults(fn=cmd_render)


# ─── doctor ──────────────────────────────────────────────────────────────────
def cmd_doctor(args) -> int:
    """Report environment and adapter readiness.  Non-zero on failures."""
    import platform
    print(f"python    : {platform.python_version()}  ({sys.executable})")
    print(f"platform  : {platform.platform()}")

    # Core deps
    print("\ncore deps:")
    for mod, pretty in (
        ("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
        ("yaml", "pyyaml"), ("spikeinterface", "spikeinterface"),
        ("probeinterface", "probeinterface"), ("neo", "neo"), ("tqdm", "tqdm"),
    ):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            print(f"  {pretty:20s} {v}")
        except Exception as e:
            print(f"  {pretty:20s} MISSING ({type(e).__name__}: {e})")

    # Optional sorters
    print("\noptional sorter packages:")
    for mod, pretty in (
        ("kilosort", "kilosort"), ("dartsort", "dartsort"),
        ("mountainsort5", "mountainsort5"), ("tridesclous", "tridesclous"),
        ("slay", "slay"), ("spikeforest", "spikeforest"),
        ("kachery_cloud", "kachery_cloud"), ("one", "one-api"),
        ("fiber_kit", "fiber_kit"),
    ):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            print(f"  {pretty:20s} {v}")
        except Exception:
            print(f"  {pretty:20s} (not installed)")

    # GPU
    print("\naccelerator:")
    try:
        import torch
        print(f"  torch                {torch.__version__}")
        print(f"  cuda.is_available    {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  cuda.device_count    {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    gpu[{i}]            {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"  torch                (not installed: {type(e).__name__})")

    # Adapters
    print("\nadapters:")
    try:
        import_registries()
        from ... import adapter_registry
    except Exception as e:
        print(f"  (registry import failed: {type(e).__name__}: {e})")
        return 1
    failures = 0
    for name in sorted(adapter_registry):
        a = adapter_registry[name]
        ok, reason = a.available()
        label = "OK" if ok else "SKIP"
        suffix = f" ({reason})" if reason else ""
        print(f"  [{label:4s}] {name:30s} v{a.version}{suffix}")
        if not ok:
            failures += 1
    print(f"\n{len(adapter_registry)} adapter(s): "
          f"{len(adapter_registry) - failures} available, {failures} unavailable")
    return 0 if failures == 0 else 0  # non-fatal: unavailable is informational


def register_doctor(subparsers) -> None:
    ap = subparsers.add_parser("doctor",
                               help="environment / dependency / adapter readiness check")
    ap.set_defaults(fn=cmd_doctor)
