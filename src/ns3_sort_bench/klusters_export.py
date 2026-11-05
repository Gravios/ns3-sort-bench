"""Export Klusters-openable sessions from benchmark results.

Each (adapter, corpus item) pair gets its own stand-alone session dir at

    <output_dir>/klusters_<adapter_slug>/<item_slug>/
        sim_session.fil                 shared raw int16 trace
        sim_session.yaml                session descriptor (+ programs block)
        sim_session.res.standard.1      THIS adapter's spike times     (int64)
        sim_session.spk.standard.1      waveforms at those times       (int16)
        sim_session.fet.standard.1      PCA features                    (int)
        sim_session.pca.standard.1      PCAE basis                      (float)
        sim_session.clu.standard.1.<raw_tag>      adapter's raw sort    (int32)
        sim_session.clu.standard.1.curated        after human curation  (int32)
        sim_session.cur.standard.1                curation log (JSON Lines)

Per-adapter top-level folders let the user curate each sorter's output
independently (open <output>/klusters_<sorter>/<item>/sim_session.yaml
in Klusters, merge the fragments, Save; Klusters writes the curated
.clu under a `curated` stage tag + a .cur action log).  Later passes
can then:

    * score the curated .clu against GT via per_cell / oracle_merge
      (`ns3sb score-curated`) to confirm parity, and
    * read the .cur log via `ns3sb curate-diff` to measure how much
      effort the curation took — comparable across sorters applied to
      the same corpus item.

The `.fil` + `.yaml` are hardlinked / copied from the item's upstream
session (the fiber-kit tmp dir when `keep_fk_artifacts=true`, or the
lab-session base for `lab:` corpora).  The adapter's `.res` is written
directly from the SortedResult; `ndm_extractspikes` + `ndm_pca` then
produce the matching `.spk`, `.fet` and `.pca`.

Ground truth exports as its OWN pseudo-sorter under
``klusters_ground_truth/<item>/``, on equal footing with every real
sorter — curate-able like any other for a sanity check, and giving
``score-curated`` a single conventional place to find the reference
clustering when the corpus's in-memory GT isn't available.
"""
from __future__ import annotations

import os
from typing import Mapping, Optional

import numpy as np

from ._base import Corpus, SortedResult
from .adapters.fiber_kit import (
    _klusters_path, _slugify,
    _NS3_METHOD, _NS_SAMPLES, _NS_PEAK,
    _RES_DTYPE, _programs_yaml_section, _run_ndm, _bandpass,
    _NDM_DETECT_DEFAULTS,
)

_CLU_DTYPE = np.dtype("<i4")


def _write_clu(path: str, cluster_ids: np.ndarray) -> None:
    """Klusters `.clu`: int32 header (n_distinct ids) + int32 body.

    Matches fiber-kit's `read_clu_file` / `writeClu` byte layout.
    """
    arr = np.ascontiguousarray(cluster_ids, dtype=_CLU_DTYPE)
    n_distinct = int(np.unique(arr).size) if arr.size else 0
    with open(path, "wb") as f:
        np.array([n_distinct], dtype=_CLU_DTYPE).tofile(f)
        arr.tofile(f)


def _flatten_sorted_result(sr: SortedResult):
    """Return `(times, cluster_ids)` one-per-spike, ascending by time."""
    times, clusters = [], []
    for u, ts in sr.spike_times.items():
        t = np.asarray(ts, dtype=np.int64)
        if t.size == 0:
            continue
        times.append(t)
        clusters.append(np.full(t.size, int(u), dtype=np.int64))
    if not times:
        return np.array([], np.int64), np.array([], np.int64)
    times = np.concatenate(times)
    clusters = np.concatenate(clusters)
    order = np.argsort(times, kind="stable")
    return times[order], clusters[order]


def _link_or_copy(src: str, dst: str) -> None:
    if os.path.exists(dst):
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def _write_yaml_with_programs(path: str, *, n_ch: int, sr: float,
                                n_samples: int = _NS_SAMPLES,
                                peak_sample: int = _NS_PEAK,
                                n_features: int = 3,
                                noise_duration_s: float = 10.0) -> None:
    """Session yaml with a `programs:` block whose rate-dependent knobs
    are rewritten for ``sr`` — mirrors adapters/fiber_kit.py's
    ``_materialise_recording``, so an exported session matches a
    sort-time session byte-for-byte on the ndm params that depend on
    the recording."""
    import yaml
    refractory_samples  = max(1, int(round(0.5e-3 * float(sr))))    # 0.5 ms
    peak_search_samples = int(n_samples)                             # snippet width
    with open(path, "w") as f:
        yaml.safe_dump({
            "parameters": {"version": "1.0",
                            "creator": "ns3-sort-bench:klusters-export"},
            "acquisitionSystem": dict(
                nBits=16, nChannels=int(n_ch),
                voltageRange=20, amplification=1000,
                offset=0, samplingRate=float(sr),
            ),
            "spikeDetection": {"channelGroups": [dict(
                channels=list(range(n_ch)),
                nSamples=int(n_samples),
                peakSampleIndex=int(peak_sample),
                nFeatures=int(n_features),
            )]},
            "programs": _programs_yaml_section(overrides={
                "ndm_detectspikes": {
                    "refractoryPeriod":  str(refractory_samples),
                    "peakSearchLength":  str(peak_search_samples),
                    "duration":          f"{noise_duration_s:g}",
                },
            }),
        }, f, sort_keys=False)


def _materialise_shared_files(recording, base: str,
                                n_samples: int = _NS_SAMPLES,
                                peak_sample: int = _NS_PEAK,
                                n_features: int = 3) -> None:
    """Idempotent: write `<base>.fil` + `<base>.dat` + `<base>.yaml` when not
    already there.  `.dat` is a hardlink to `.fil` — Klusters' TracesProvider
    is hardcoded to open `<base>.dat` (src/klusters/src/klustersdoc.cpp:745)
    while fiber-session reads `.fil`.  Same bytes, no duplicate on disk.

    The traces are bandpass-filtered 300-6000 Hz before being written, so
    the exported .fil matches what ndm_detectspikes expects (a hipassed
    signal) and reproduces adapters/fiber_kit.py's _materialise_recording
    byte-for-byte.
    """
    fil_path = base + ".fil"
    dat_path = base + ".dat"
    yaml_path = base + ".yaml"
    if os.path.exists(fil_path) and os.path.exists(dat_path) \
            and os.path.exists(yaml_path):
        return
    n_samp = int(recording.get_num_samples(0))
    n_ch = int(recording.get_num_channels())
    sr = float(recording.get_sampling_frequency())
    traces = recording.get_traces(start_frame=0, end_frame=n_samp)
    traces_filt = _bandpass(np.asarray(traces, np.float64), sr)
    np.clip(np.rint(traces_filt), -32768, 32767).astype(np.int16).tofile(fil_path)
    _link_or_copy(fil_path, dat_path)
    noise_s = min(float(_NDM_DETECT_DEFAULTS.get("duration", "10")),
                   (n_samp / sr) * 0.5)
    _write_yaml_with_programs(
        yaml_path, n_ch=n_ch, sr=sr,
        n_samples=n_samples, peak_sample=peak_sample,
        n_features=n_features, noise_duration_s=noise_s,
    )


def export_adapter_session(output_dir: str, corpus: Corpus,
                             adapter_name: str, candidate: SortedResult,
                             *, group: int = 1,
                             n_samples: int = _NS_SAMPLES,
                             peak_sample: int = _NS_PEAK,
                             method: str = _NS3_METHOD,
                             shared_from: Optional[str] = None) -> str:
    """Write a stand-alone Klusters session for one adapter's result.

    Writes .fil + .yaml + .res + .clu directly; runs `ndm_extractspikes`
    and `ndm_pca` on the generated yaml to produce .spk + .fet + .pca.

    `shared_from` names the ``<base>`` of a session dir already populated
    with `.fil` / `.dat` / `.yaml` — when given, this call hardlinks those
    files into the new session dir instead of writing a second on-disk copy.
    Caller (``export_klusters_session``) uses this to dedupe the identical
    .fil across every sorter's session dir on a full-sweep run.

    Returns the session directory, suitable for ``klusters <dir>/sim_session.yaml``.
    """
    item_slug = _slugify(corpus.name)
    adapter_slug = _slugify(adapter_name)
    sess_dir = os.path.join(output_dir, f"klusters_{adapter_slug}", item_slug)
    os.makedirs(sess_dir, exist_ok=True)
    base = os.path.join(sess_dir, "sim_session")

    # Shared .fil + .dat + .yaml.  Three sources, in order of preference:
    #
    #   1. `shared_from` — an earlier session dir in this same export pass
    #      (the shared-files dedup across sorters).  Hardlinks all three
    #      files from there instead of rebuilding them.
    #   2. A lab session's existing .fil/.dat/.yaml, when the recording
    #      came from one (annotated as `session_base`).  Hardlinks from
    #      the lab path.
    #   3. Build them from the SpikeInterface recording in-memory — the
    #      fully from-scratch case.
    #
    # Klusters reads <base>.dat; fiber-session reads <base>.fil.  Either
    # hardlink path mirrors adapters/fiber_kit.py::_materialise_recording's
    # ".dat ↔ .fil hardlink" so both consumers find what they expect.
    recording = corpus.recording
    sess_base = recording.get_annotation("session_base", None) \
                if hasattr(recording, "get_annotation") else None
    if shared_from and (os.path.exists(shared_from + ".fil")
                         or os.path.exists(shared_from + ".dat")):
        src_fil = shared_from + ".fil" if os.path.exists(shared_from + ".fil") \
                  else shared_from + ".dat"
        _link_or_copy(src_fil, base + ".fil")
        _link_or_copy(src_fil, base + ".dat")
        if os.path.exists(shared_from + ".yaml"):
            _link_or_copy(shared_from + ".yaml", base + ".yaml")
        else:
            _write_yaml_with_programs(
                base + ".yaml",
                n_ch=recording.get_num_channels(),
                sr=recording.get_sampling_frequency(),
                n_samples=n_samples, peak_sample=peak_sample,
            )
    elif sess_base and (os.path.exists(sess_base + ".fil")
                         or os.path.exists(sess_base + ".dat")):
        src = sess_base + ".fil" if os.path.exists(sess_base + ".fil") \
              else sess_base + ".dat"
        _link_or_copy(src, base + ".fil")
        _link_or_copy(src, base + ".dat")
        if os.path.exists(sess_base + ".yaml"):
            _link_or_copy(sess_base + ".yaml", base + ".yaml")
        else:
            _write_yaml_with_programs(
                base + ".yaml",
                n_ch=recording.get_num_channels(),
                sr=recording.get_sampling_frequency(),
                n_samples=n_samples, peak_sample=peak_sample,
            )
    else:
        _materialise_shared_files(recording, base,
                                    n_samples=n_samples,
                                    peak_sample=peak_sample)

    # Per-adapter .res + .clu written directly from the SortedResult.
    peaks, cluster_ids = _flatten_sorted_result(candidate)
    # Boundary filter matching fiber-kit's extract-snippets behaviour: a spike
    # too close to either end of the trace yields no valid waveform, so drop
    # it from both .res and .clu to keep them row-aligned.
    n_samp = int(recording.get_num_samples(0))
    before, after = peak_sample, n_samples - peak_sample
    ok = (peaks >= before) & (peaks + after <= n_samp)
    kept_peaks = np.sort(peaks[ok].astype(_RES_DTYPE))
    kept_clu = cluster_ids[ok].astype(_CLU_DTYPE)
    # Keep .res / .clu row-aligned after the sort by (time) order
    order = np.argsort(peaks[ok], kind="stable")
    kept_clu = cluster_ids[ok][order].astype(_CLU_DTYPE)

    res_path = _klusters_path(base, "res", group, method=method)
    clu_path = _klusters_path(base, "clu", group, method=method)
    with open(res_path, "wb") as f:
        kept_peaks.tofile(f)
    _write_clu(clu_path, kept_clu)

    # Extract waveforms + features for THIS adapter's spike times via ndm_*.
    session_yaml = base + ".yaml"
    log_dir = sess_dir
    _run_ndm("ndm_extractspikes", session_yaml,
              extra=["--method", method],
              log_path=os.path.join(log_dir, "ndm_extractspikes.log"))
    _run_ndm("ndm_pca", session_yaml,
              extra=["--method", method],
              log_path=os.path.join(log_dir, "ndm_pca.log"))
    return sess_dir


def export_klusters_session(output_dir: str, corpus: Corpus,
                              candidates: Mapping[str, SortedResult],
                              *, include_ground_truth: bool = True,
                              **kw) -> list[str]:
    """Write per-adapter Klusters sessions for one corpus item.

    `candidates` maps adapter name → SortedResult.  If `include_ground_truth`
    is true and the corpus has a non-empty ground truth, it is exported as
    a pseudo-adapter under ``klusters_ground_truth/<item>/`` on equal
    footing with the real sorters.

    Shared-files dedup: the ``.fil`` / ``.dat`` / ``.yaml`` carried by every
    per-adapter session are byte-identical across sorters for a given corpus
    item — only the ``.res`` / ``.clu`` / ``.spk`` / ``.fet`` / ``.pca``
    differ per sort.  Materialising .fil once and hardlinking it into each
    adapter's session dir costs one on-disk copy total instead of one per
    sorter (40 sorters × a few hundred MB per .fil is where this starts to
    matter).  The first adapter to export writes the real file; every later
    adapter hardlinks it, falling back to a copy on cross-filesystem tmp.
    Returns the list of per-adapter session dirs written.
    """
    results = dict(candidates)
    if include_ground_truth and corpus.ground_truth is not None \
            and "ground_truth" not in results:
        results["ground_truth"] = corpus.ground_truth
    adapter_names = list(results)

    # Materialise the shared .fil/.dat/.yaml once into the FIRST adapter's
    # session dir, then pass that dir to the rest of the exports as a
    # hardlink source.  Preserves per-adapter session self-containedness
    # (every dir still has its own file names) without n copies of the
    # bytes.
    anchor_base: Optional[str] = None
    out_dirs: list[str] = []
    for name in adapter_names:
        anchor = anchor_base    # None for the first, populated afterwards
        sess_dir = export_adapter_session(output_dir, corpus, name,
                                            results[name],
                                            shared_from=anchor, **kw)
        out_dirs.append(sess_dir)
        if anchor_base is None:
            anchor_base = os.path.join(sess_dir, "sim_session")
    return out_dirs
