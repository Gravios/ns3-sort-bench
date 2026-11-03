"""Contract system tests — the new piece added by the standalone-project
port.  Checks:
    - every adapter declares a Contract instance
    - every Contract validates (dotted-path yaml field access works)
    - validate_yaml returns reasons for missing required fields
    - validate_recording returns reasons for failed preconditions
    - to_markdown emits a non-empty section with the required headings
"""
import pytest
import ns3_sort_bench.adapters   # noqa: F401 — populate registry
from ns3_sort_bench import adapter_registry
from ns3_sort_bench.contracts import (
    Contract, RecordingReqs, OutputSchema, Capabilities, _dotted_get,
)


def test_every_adapter_has_contract():
    assert len(adapter_registry) > 0
    for name, a in adapter_registry.items():
        assert a.contract is not None, f"{name}: no contract declared"
        assert a.contract.adapter_name, f"{name}: contract has empty adapter_name"
        assert a.contract.upstream_url.startswith("http"), \
            f"{name}: upstream_url looks wrong: {a.contract.upstream_url!r}"


def test_dotted_get_handles_lists():
    d = {"a": {"b": [{"c": 42}, {"c": 43}]}}
    assert _dotted_get(d, "a.b[0].c") == 42
    assert _dotted_get(d, "a.b[1].c") == 43
    with pytest.raises((KeyError, IndexError)):
        _dotted_get(d, "a.b[9].c")


def test_validate_yaml_catches_missing_required():
    c = Contract(
        adapter_name="test", upstream_url="https://x", upstream_version_pin="0",
        yaml_fields_required=("acquisitionSystem.samplingRate", "spikeDetection.channelGroups"),
        yaml_fields_optional=(),
    )
    # Missing both
    ok, reasons = c.validate_yaml({})
    assert not ok and len(reasons) == 2
    # One present
    ok, reasons = c.validate_yaml({"acquisitionSystem": {"samplingRate": 20000}})
    assert not ok and len(reasons) == 1
    # Both present
    ok, reasons = c.validate_yaml({
        "acquisitionSystem": {"samplingRate": 20000},
        "spikeDetection": {"channelGroups": [{"channels": [0, 1]}]},
    })
    assert ok and reasons == []


def test_validate_recording_catches_too_few_channels():
    class Rec:
        def get_num_channels(self): return 1
        def get_sampling_frequency(self): return 20000.0
        def get_num_samples(self, i): return 100_000
    c = Contract(
        adapter_name="test", upstream_url="https://x", upstream_version_pin="0",
        yaml_fields_required=(), yaml_fields_optional=(),
        recording=RecordingReqs(min_channels=8),
    )
    ok, reasons = c.validate_recording(Rec())
    assert not ok and "min 8" in reasons[0]


def test_contract_to_markdown_renders_sections():
    a = next(iter(adapter_registry.values()))
    md = a.contract.to_markdown()
    assert "###" in md
    assert "Upstream" in md
    assert "Version pin" in md
    assert "Signal space" in md


def test_every_contract_has_known_signal_space():
    valid_spaces = {"raw", "filtered", "stderiv", "whitened"}
    for name, a in adapter_registry.items():
        assert a.contract.recording.signal_space in valid_spaces, (
            f"{name}: signal_space={a.contract.recording.signal_space!r} not in {valid_spaces}"
        )


def test_every_contract_has_known_overlap_rule():
    valid_rules = {"none", "iterative_subtract", "sic", "omp"}
    for name, a in adapter_registry.items():
        assert a.contract.capabilities.overlap_resolution in valid_rules, (
            f"{name}: overlap_resolution={a.contract.capabilities.overlap_resolution!r}"
        )
