"""`ns3sb contract <adapter>` — print one adapter's full Contract."""
from __future__ import annotations

from .._util import eprint, import_registries


def cmd_contract(args) -> int:
    import_registries()
    from ... import adapter_registry
    if args.adapter not in adapter_registry:
        eprint(f"unknown adapter: {args.adapter!r}")
        eprint(f"  available: {sorted(adapter_registry)}")
        return 1
    c = adapter_registry[args.adapter].contract
    if c is None:
        eprint(f"{args.adapter}: no Contract declared")
        return 1
    if args.format == "markdown" and hasattr(c, "to_markdown"):
        print(c.to_markdown())
        return 0
    if args.format == "yaml":
        try:
            import yaml
        except ImportError:
            eprint("pyyaml required for --format yaml")
            return 1
        from dataclasses import asdict, is_dataclass
        payload = asdict(c) if is_dataclass(c) else {
            k: getattr(c, k) for k in dir(c) if not k.startswith("_")
        }
        print(yaml.safe_dump(payload, sort_keys=False))
        return 0
    # text (default)
    print(f"adapter: {args.adapter}")
    print(f"  required_yaml: {list(c.yaml_fields_required)}")
    print(f"  optional_yaml: {list(c.yaml_fields_optional)}")
    print(f"  recording:     {c.recording}")
    print(f"  capabilities:  {c.capabilities}")
    print(f"  output:        {c.output}")
    print(f"  resources:     {c.resources}")
    return 0


def register(subparsers) -> None:
    ap = subparsers.add_parser("contract", help="print one adapter's Contract")
    ap.add_argument("adapter", help="e.g. fiber_kit, botm_ref, kilosort4")
    ap.add_argument("--format", choices=("text", "markdown", "yaml"), default="text")
    ap.set_defaults(fn=cmd_contract)
