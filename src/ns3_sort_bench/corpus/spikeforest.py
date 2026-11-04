"""SpikeForest corpus loader (Flatiron Institute, Magland et al. 2020).

Uses the upstream `spikeforest` Python package (0.12.x) directly rather than
SpikeInterface's extractor shim — the latter has moved across SI releases.
The upstream package returns a MDA-backed `BaseRecording` plus a `BaseSorting`
of the ground-truth spikes; we wrap those into our `Corpus` + `SortedResult`.

Paired study sets (ground truth = intracellular/juxtacellular reference cell):

    PAIRED_CRCNS_HC1     rat hippocampus, tetrode + intracellular   (Buzsaki)
    PAIRED_KAMPFF        mouse cortex, 32-of-N probe + patch        (Kampff)
    PAIRED_ENGLISH       awake mouse, 4-32 ch + juxtacellular       (English)
    PAIRED_BOYDEN        mouse cortex, 32-of-N probe + patch        (Boyden)
    PAIRED_MEA64C_YGER   mouse retina, 64-of-252 MEA + patch        (Yger)
    PAIRED_MONOTRODE     1-ch sub-selected from the four above

Synthetic study sets (ground truth = known injected spikes):

    SYNTH_MEAREC_TETRODE / _NEURONEXUS / _NP16 / _NP32 ...
    SYNTH_MONOTRODE / SYNTH_MAGLAND / SYNTH_VISAPY ...

For the simultaneous-whole-cell use case the user asked for, PAIRED_CRCNS_HC1
is the anatomical match (rat CA1); PAIRED_KAMPFF has the best signal-to-noise.

The spikeforest package needs kachery-cloud for the first-time pull of each
recording's raw bytes.  First call authenticates against kachery-cloud; later
calls hit the local cache.  Set env `KACHERY_CLOUD_DIR` to control cache
location.

Install:   pip install ns3-sort-bench[spikeforest]
"""
from __future__ import annotations

import os
from typing import Iterable, Iterator, List, Optional, Sequence

import numpy as np

from .._base import Corpus, SortedResult


# ─── study-set metadata ──────────────────────────────────────────────────────
#
# Hard-coded because the upstream recordings.json is small and these are the
# fields the harness needs for display / contract-validation.  Pulled from
# Magland et al. 2020 Table 2.
PAIRED_STUDY_SETS = {
    "PAIRED_CRCNS_HC1":    dict(species="rat",   region="hippocampus", gt="intracellular"),
    "PAIRED_KAMPFF":       dict(species="mouse", region="cortex",       gt="patch"),
    "PAIRED_ENGLISH":      dict(species="mouse", region="cortex/hc/th", gt="juxtacellular"),
    "PAIRED_BOYDEN":       dict(species="mouse", region="cortex",       gt="patch"),
    "PAIRED_MEA64C_YGER":  dict(species="mouse", region="retina",       gt="patch"),
    "PAIRED_MONOTRODE":    dict(species="various", region="various",    gt="paired-subset"),
}

SYNTH_STUDY_SETS = {
    "SYNTH_MEAREC_TETRODE",
    "SYNTH_MEAREC_NEURONEXUS",
    "SYNTH_MEAREC_NP16",
    "SYNTH_MEAREC_NP32",
    "SYNTH_MONOTRODE",
    "SYNTH_MAGLAND",
    "SYNTH_VISAPY",
    "SYNTH_BIONET",
}


# ─── lazy upstream import ────────────────────────────────────────────────────
def _import_spikeforest():
    try:
        import spikeforest as sf
    except ImportError as e:
        raise ImportError(
            "The spikeforest package is required.  Install with\n"
            "    pip install 'ns3-sort-bench[spikeforest]'\n"
            "(upstream: https://github.com/flatironinstitute/spikeforest)"
        ) from e
    try:
        import kachery_cloud as kcl  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "kachery_cloud is required by spikeforest.  Install with\n"
            "    pip install 'ns3-sort-bench[spikeforest]'"
        ) from e
    return sf


# ─── listing ─────────────────────────────────────────────────────────────────
def list_recordings(study_set: Optional[str] = None,
                    study_name: Optional[str] = None,
                    uri: Optional[str] = None) -> List[dict]:
    """List available SpikeForest recordings, optionally filtered.

    Returns a list of dicts with keys:
        study_set_name, study_name, recording_name,
        sampling_frequency, num_channels, duration_sec, num_true_units

    First call downloads the recordings-index JSON (~1 MB) via kachery.
    """
    sf = _import_spikeforest()
    all_recs = sf.load_spikeforest_recordings() if uri is None else \
               sf.load_spikeforest_recordings(uri)
    out = []
    for R in all_recs:
        if study_set is not None and R.study_set_name != study_set:
            continue
        if study_name is not None and R.study_name != study_name:
            continue
        out.append(dict(
            study_set_name=R.study_set_name,
            study_name=R.study_name,
            recording_name=R.recording_name,
            sampling_frequency=R.sampling_frequency,
            num_channels=R.num_channels,
            duration_sec=R.duration_sec,
            num_true_units=R.num_true_units,
        ))
    return out


def _find_recording(study_name: str, recording_name: str,
                    uri: Optional[str] = None):
    sf = _import_spikeforest()
    kwargs = dict(study_name=study_name, recording_name=recording_name, uri=uri)
    return sf.load_spikeforest_recording(**kwargs)


# ─── Corpus wrapping ─────────────────────────────────────────────────────────
# ns3 / Klusters conventions reserve unit IDs 0 (artefact) and 1 (noise); real
# units land at >= 2, and every reserved-ID-aware scorer drops those.
# SpikeForest paired recordings use unit 1 as the patched reference cell, and
# other sorting-truth stores commonly start at 0 or 1 too — the whole GT would
# vanish on dropped_reserved().  Shift every upstream GT unit id by this offset
# so the Klusters-convention filter sees them as real units.  SortedResult
# records the offset in .extras so a diagnostic can reverse it if needed.
_GT_UNIT_ID_OFFSET = 2


def _sorting_true_to_sorted_result(sorting_ext, *, corpus_item: str,
                                    sampling_rate: float,
                                    unit_id_offset: int = _GT_UNIT_ID_OFFSET
                                    ) -> SortedResult:
    """Convert a SpikeInterface BaseSorting (ground truth) → SortedResult.

    For paired recordings, SpikeForest ground truth is typically ONE unit
    (the simultaneously-patched cell).  For synthetic ones, it's all injected
    cells.  Either way, the shape is: unit_ids → spike-sample arrays.

    Every unit id is shifted by ``unit_id_offset`` (default 2) so the Klusters
    reserved-ID convention (0 = artefact, 1 = noise) does not swallow the
    real GT units.  The original SpikeForest id is kept per unit in
    ``extras['unit_id_sf_original']`` as `{shifted_id: original_id}`.
    """
    unit_ids_raw = list(sorting_ext.get_unit_ids())
    spike_times = {}
    original_by_shifted = {}
    for u in unit_ids_raw:
        try:
            train = sorting_ext.get_unit_spike_train(u)
        except TypeError:
            # Some SI versions want segment_index=0 explicitly
            train = sorting_ext.get_unit_spike_train(u, segment_index=0)
        shifted = int(u) + int(unit_id_offset)
        spike_times[shifted] = np.asarray(train, dtype=np.int64)
        original_by_shifted[shifted] = int(u)
    return SortedResult(
        unit_ids=sorted(spike_times),
        spike_times=spike_times,
        sampling_rate=float(sampling_rate),
        adapter="ground_truth",
        corpus_item=corpus_item,
        adapter_version="spikeforest",
        extras={
            "unit_id_offset": int(unit_id_offset),
            "unit_id_sf_original": original_by_shifted,
        },
    )


def load(study_name: str, recording_name: str,
         *, uri: Optional[str] = None,
         ns3_yaml: Optional[dict] = None,
         corpus_kind: str = "spikeforest") -> Corpus:
    """Load one SpikeForest item as a Corpus.

    Parameters
    ----------
    study_name : e.g. 'paired_crcns_hc1_d533101', 'paired_kampff', 'SYNTH_MEAREC_TETRODE_noise10_K20_C8'
    recording_name : the item identifier within the study
    uri : optional override for the recordings-index JSON kachery URI
    ns3_yaml : optional parsed ns3 session yaml (for contract validation)

    Returns
    -------
    Corpus with
        recording       — SpikeInterface BaseRecording (MDA-backed)
        ground_truth    — SortedResult built from the paired/synthetic truth
        kind            — 'spikeforest' (unless overridden)
        description     — species / region / ground-truth type / channel count
    """
    R = _find_recording(study_name=study_name, recording_name=recording_name,
                        uri=uri)
    recording = R.get_recording_extractor()
    sorting_true = R.get_sorting_true_extractor()

    item_name = f"{R.study_set_name}:{R.study_name}:{R.recording_name}"
    gt = _sorting_true_to_sorted_result(
        sorting_true, corpus_item=item_name,
        sampling_rate=R.sampling_frequency,
    )

    meta = PAIRED_STUDY_SETS.get(R.study_set_name, dict(species="?", region="?", gt="?"))
    description = (
        f"{R.study_set_name} ({meta['species']}, {meta['region']}, "
        f"gt={meta['gt']}) · {R.num_channels} ch · "
        f"{R.sampling_frequency / 1000:.1f} kHz · "
        f"{R.duration_sec:.1f} s · {R.num_true_units} true unit(s)"
    )

    return Corpus(
        name=item_name,
        recording=recording,
        ground_truth=gt,
        description=description,
        kind=corpus_kind,
        ns3_yaml=ns3_yaml,
        extras=dict(
            study_set_name=R.study_set_name,
            study_name=R.study_name,
            recording_name=R.recording_name,
            source="spikeforest",
            spikeforest_record=R.recording_record,
        ),
    )


def iter_study_set(study_set: str, *, uri: Optional[str] = None,
                   limit: Optional[int] = None) -> Iterator[Corpus]:
    """Yield Corpus for every recording in a study set.

    Convenient for sweeping an entire paired set in one BenchmarkRun:

        for c in iter_study_set("PAIRED_CRCNS_HC1", limit=5):
            bench.add_corpus(c)
    """
    recs = list_recordings(study_set=study_set, uri=uri)
    if limit is not None:
        recs = recs[:limit]
    for rec in recs:
        yield load(study_name=rec["study_name"],
                   recording_name=rec["recording_name"],
                   uri=uri)


# ─── CLI helper ──────────────────────────────────────────────────────────────
def _cli_list(argv: Sequence[str]) -> int:
    """`python -m ns3_sort_bench.corpus.spikeforest list --study-set PAIRED_CRCNS_HC1`"""
    import argparse
    ap = argparse.ArgumentParser(description="List SpikeForest recordings")
    ap.add_argument("--study-set", default=None,
                    help="e.g. PAIRED_CRCNS_HC1, PAIRED_KAMPFF, SYNTH_MEAREC_TETRODE")
    ap.add_argument("--study-name", default=None,
                    help="narrow to a specific study within the set")
    args = ap.parse_args(argv)
    rows = list_recordings(study_set=args.study_set, study_name=args.study_name)
    if not rows:
        print(f"(no recordings matched: study_set={args.study_set}, study_name={args.study_name})")
        return 1
    hdr = ("study_set_name", "study_name", "recording_name",
           "sampling_frequency", "num_channels", "duration_sec", "num_true_units")
    print("\t".join(hdr))
    for r in rows:
        print("\t".join(str(r[k]) for k in hdr))
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "list":
        sys.exit(_cli_list(sys.argv[2:]))
    print(__doc__)
    sys.exit(0)
