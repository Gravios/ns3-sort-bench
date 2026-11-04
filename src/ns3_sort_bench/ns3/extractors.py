"""SpikeInterface extractors for the neurosuite-3 session format.

- NS3RecordingExtractor   : raw .fil / .dat memmap as a SI BaseRecording
- NS3SortingExtractor     : one group's (.res.N, .clu.N) as a SI BaseSorting
- NS3MultiGroupSorting    : all groups → one sorting with offset-remapped IDs
- read_ns3_session        : one-call entry — {group: (recording, sorting)}

Delegates to fiber_kit.neuro_io for variant/tag-aware .res/.clu paths when
fiber_kit is installed; falls back to canonical paths otherwise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from spikeinterface.core import (
    BaseRecording, BaseRecordingSegment, BaseSorting, BaseSortingSegment,
)

from .session import NS3Session, load_session_yaml

try:
    from fiber_kit import neuro_io as _fk_io
    from fiber_kit import fiber_lib as _fk_lib
    HAVE_FK = True
except ImportError:
    HAVE_FK = False


_DAT_EXTS = ("fil", "dat")


def _resolve_raw_path(base: str, prefer: Sequence[str] = _DAT_EXTS) -> Tuple[str, str]:
    for ext in prefer:
        p = f"{base}.{ext}"
        if os.path.exists(p):
            return p, ext
    raise FileNotFoundError(
        f"{base}: neither .{prefer[0]} nor .{prefer[-1]} exists")


class _NS3RecordingSegment(BaseRecordingSegment):
    def __init__(self, memmap: np.memmap, channel_indices: np.ndarray,
                 sampling_frequency: float, dtype: np.dtype,
                 transform: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        BaseRecordingSegment.__init__(self, sampling_frequency=sampling_frequency)
        self._mm = memmap
        self._ch = np.asarray(channel_indices, dtype=np.int64)
        self._dtype = np.dtype(dtype)
        self._transform = transform
        self._n = int(memmap.shape[0])

    def get_num_samples(self) -> int:
        return self._n

    def get_traces(self, start_frame=None, end_frame=None, channel_indices=None):
        s0 = 0 if start_frame is None else int(start_frame)
        s1 = self._n if end_frame is None else int(end_frame)
        if s1 < s0:
            raise ValueError(f"end_frame ({s1}) < start_frame ({s0})")

        contiguous = self._ch.size > 0 and bool((np.diff(self._ch) == 1).all())
        lo, hi = int(self._ch[0]), int(self._ch[-1]) + 1
        if contiguous:
            win = self._mm[s0:s1, lo:hi]
        else:
            win = self._mm[s0:s1][:, self._ch]

        if self._transform is not None:
            win = self._transform(np.asarray(win, np.float64))

        if channel_indices is not None:
            if isinstance(channel_indices, slice):
                win = win[:, channel_indices]
            else:
                win = win[:, np.asarray(channel_indices, np.int64)]
        return np.ascontiguousarray(win, dtype=self._dtype)


class NS3RecordingExtractor(BaseRecording):
    """SI recording over an ns3 raw signal file (.fil or .dat)."""
    extractor_name = "ns3"
    is_writable = False
    mode = "file"
    installation_mesg = ""

    def __init__(self, session_yaml, channels=None, raw="fil",
                 transform=None, gain_to_uV=None, offset_to_uV=None,
                 probe_geometry=None, name="ns3"):
        self.sess = load_session_yaml(session_yaml)

        if os.path.sep in str(raw) or os.path.isabs(str(raw)):
            raw_path = str(raw)
        else:
            raw_path, _ = _resolve_raw_path(
                self.sess.base, prefer=(str(raw),) + _DAT_EXTS)

        raw_dtype = np.int16 if self.sess.n_bits <= 16 else np.int32
        nbytes = os.path.getsize(raw_path)
        bytes_per_sample = raw_dtype().itemsize * self.sess.n_channels_total
        if nbytes % bytes_per_sample:
            raise ValueError(
                f"{raw_path}: size {nbytes} is not a multiple of "
                f"{bytes_per_sample} bytes/sample")
        n_samp = nbytes // bytes_per_sample
        mm = np.memmap(raw_path, dtype=raw_dtype, mode="r",
                       shape=(n_samp, self.sess.n_channels_total))

        if channels is None:
            channels = np.arange(self.sess.n_channels_total, dtype=np.int64)
        else:
            channels = np.asarray(list(channels), dtype=np.int64)
        channel_ids = [str(int(c)) for c in channels]

        if gain_to_uV is None:
            gain_to_uV = (self.sess.voltage_range_mV * 1e3) / \
                         (self.sess.amplification * (2 ** self.sess.n_bits))
        if offset_to_uV is None:
            offset_to_uV = 0.0

        out_dtype = np.float32 if transform is not None else raw_dtype

        BaseRecording.__init__(
            self, sampling_frequency=self.sess.sampling_rate,
            channel_ids=channel_ids, dtype=out_dtype)
        self.add_recording_segment(_NS3RecordingSegment(
            memmap=mm, channel_indices=channels,
            sampling_frequency=self.sess.sampling_rate,
            dtype=out_dtype, transform=transform,
        ))

        n_ch_out = len(channel_ids)
        self.set_channel_gains(np.full(n_ch_out, gain_to_uV, dtype=np.float32))
        self.set_channel_offsets(np.full(n_ch_out, offset_to_uV, dtype=np.float32))

        if probe_geometry is not None:
            locs = np.asarray(probe_geometry, np.float32)
        else:
            locs = np.column_stack([
                np.zeros(n_ch_out, np.float32),
                20.0 * np.arange(n_ch_out, dtype=np.float32),
            ])
        if hasattr(self, "set_dummy_probe_from_locations"):
            self.set_dummy_probe_from_locations(locs)
        else:
            self.set_channel_locations(locs)

        self.annotate(session_base=self.sess.base, raw_path=raw_path,
                      transform=transform.__name__ if transform else None,
                      name=name)
        self._kwargs = dict(
            session_yaml=str(session_yaml), channels=channels.tolist(),
            raw=raw, transform=None if transform is None else transform.__name__,
            gain_to_uV=gain_to_uV, offset_to_uV=offset_to_uV,
            probe_geometry=None if probe_geometry is None else np.asarray(probe_geometry).tolist(),
            name=name,
        )


# ─── sorting ─────────────────────────────────────────────────────────────────
_RES_DTYPE = np.int64
_CLU_DTYPE = np.int32


def _read_res_file(path: str) -> np.ndarray:
    if HAVE_FK:
        return _fk_io.read_res_file(path)
    return np.fromfile(path, dtype=_RES_DTYPE).astype(np.int64)


def _read_clu_file(path: str, n_spikes: Optional[int] = None):
    if HAVE_FK:
        return _fk_io.read_clu_file(path, n_spikes=n_spikes)
    raw = np.fromfile(path, dtype=_CLU_DTYPE)
    n_clu = int(raw[0]) if raw.size else 0
    ids = raw[1:].astype(np.int64)
    if n_spikes is not None and ids.size != n_spikes:
        raise ValueError(f"{path}: {ids.size} labels for {n_spikes} spikes")
    return n_clu, ids


def _resolve_sort_pair(base: str, elec: int, variant: str = "", tag: str = ""):
    """Resolve (`.res`, `.clu`) for one (variant, stage-tag) sort.

    Per neurosuite-3's canonical naming (doc/ndmanager-plugins/formats/naming.md):

      * `.clu` is **MethodSpecific** — strict `<base>.clu.<variant>.<group>[.<tag>]`.
      * `.res` is **Shared** — one per group, shared across stages.  A stage
        that reextracts (realign/dedup) writes a staged copy at
        `<base>.res.<variant>.<group>.<tag>`; otherwise the shared
        `<base>.res.<variant>.<group>` (no tag) is the one that pairs with the
        tagged `.clu`.

    Resolution: try the staged `.res` first; fall back to the shared `.res` if
    the tagged copy isn't on disk.  Mirrors fiber-kit's own `read_res_at` /
    `read_res` split — staged is strict, shared is a `resolve_any`.
    """
    if HAVE_FK:
        clu = _fk_io.session_path(base, "clu", elec, variant=variant, tag=tag)
        res_staged = _fk_io.session_path(base, "res", elec, variant=variant, tag=tag)
        res_shared = _fk_io.session_path(base, "res", elec, variant=variant)
    else:
        def _p(t, with_tag):
            pfx = f"{base}.{t}"
            if variant: pfx += f".{variant}"
            pfx += f".{elec}"
            if with_tag and tag: pfx += f".{tag}"
            return pfx
        clu = _p("clu", with_tag=True)
        res_staged = _p("res", with_tag=True)
        res_shared = _p("res", with_tag=False)

    if os.path.exists(res_staged):
        res = res_staged
    elif os.path.exists(res_shared):
        res = res_shared
    else:
        raise FileNotFoundError(
            f"no .res for ({base}, group={elec}, variant={variant!r}, tag={tag!r}); "
            f"tried staged={res_staged!r} and shared={res_shared!r}"
        )
    if not os.path.exists(clu):
        raise FileNotFoundError(clu)
    return res, clu


class _NS3SortingSegment(BaseSortingSegment):
    def __init__(self, times_by_unit: Dict[int, np.ndarray]):
        BaseSortingSegment.__init__(self)
        self._tbu = {int(u): np.asarray(t, dtype=np.int64) for u, t in times_by_unit.items()}

    def get_unit_spike_train(self, unit_id, start_frame=None, end_frame=None):
        t = self._tbu.get(int(unit_id), np.array([], dtype=np.int64))
        if start_frame is None and end_frame is None:
            return t
        s0 = 0 if start_frame is None else int(start_frame)
        s1 = t[-1] + 1 if end_frame is None else int(end_frame)
        return t[np.searchsorted(t, s0, side="left"):np.searchsorted(t, s1, side="left")]


class NS3SortingExtractor(BaseSorting):
    extractor_name = "ns3"
    is_writable = False
    mode = "file"
    installation_mesg = ""

    def __init__(self, session_yaml, electrode_group: int,
                 variant: str = "", tag: str = "",
                 exclude_reserved: bool = True,
                 unit_id_offset: int = 0,
                 sampling_rate: Optional[float] = None,
                 name: str = "ns3-sort"):
        sess = load_session_yaml(session_yaml)
        res_path, clu_path = _resolve_sort_pair(
            sess.base, electrode_group, variant=variant, tag=tag)
        res = _read_res_file(res_path)
        _, clu = _read_clu_file(clu_path, n_spikes=res.size)
        sr = float(sampling_rate) if sampling_rate else sess.sampling_rate

        kept_ids = np.unique(clu)
        if exclude_reserved:
            kept_ids = kept_ids[kept_ids >= 2]
        times_by_unit = {}
        for uid in kept_ids:
            t = np.sort(res[clu == uid])
            times_by_unit[int(uid) + int(unit_id_offset)] = t

        unit_ids = [int(u) for u in sorted(times_by_unit)]
        BaseSorting.__init__(self, sampling_frequency=sr, unit_ids=unit_ids)
        self.add_sorting_segment(_NS3SortingSegment(times_by_unit))

        self.annotate(session_base=sess.base, electrode_group=int(electrode_group),
                      variant=variant, tag=tag, name=name,
                      excluded_reserved=exclude_reserved)
        self._kwargs = dict(
            session_yaml=str(session_yaml), electrode_group=int(electrode_group),
            variant=variant, tag=tag, exclude_reserved=exclude_reserved,
            unit_id_offset=int(unit_id_offset), sampling_rate=sr, name=name,
        )


class NS3MultiGroupSorting(BaseSorting):
    """All spike-detection groups → one sorting with offset-remapped IDs."""
    extractor_name = "ns3-multigroup"
    is_writable = False
    mode = "file"
    installation_mesg = ""

    def __init__(self, session_yaml, variant: str = "", tag: str = "",
                 exclude_reserved: bool = True,
                 groups: Optional[Iterable[int]] = None,
                 stride: int = 10_000, name: str = "ns3-session"):
        sess = load_session_yaml(session_yaml)
        if groups is None:
            groups = range(1, len(sess.spike_groups) + 1)
        groups = list(groups)

        combined: Dict[int, np.ndarray] = {}
        used_groups: List[int] = []
        for g in groups:
            try:
                single = NS3SortingExtractor(
                    session_yaml, electrode_group=g,
                    variant=variant, tag=tag,
                    exclude_reserved=exclude_reserved,
                    unit_id_offset=g * stride,
                    sampling_rate=sess.sampling_rate,
                )
            except FileNotFoundError:
                continue
            seg = single._sorting_segments[0]       # type: ignore[attr-defined]
            combined.update(seg._tbu)               # type: ignore[attr-defined]
            used_groups.append(g)

        unit_ids = sorted(combined)
        BaseSorting.__init__(self, sampling_frequency=sess.sampling_rate,
                             unit_ids=unit_ids)
        self.add_sorting_segment(_NS3SortingSegment(combined))
        self.annotate(session_base=sess.base, variant=variant, tag=tag,
                      ns3_group_stride=int(stride), ns3_groups_included=used_groups,
                      name=name)
        self._kwargs = dict(
            session_yaml=str(session_yaml), variant=variant, tag=tag,
            exclude_reserved=exclude_reserved,
            groups=list(groups), stride=int(stride), name=name,
        )


def read_ns3_session(session_yaml, *, variant: str = "", tag: str = "",
                     raw: str = "fil", transform: Optional[Callable] = None,
                     exclude_reserved: bool = True):
    """Load a session as per-group (recording, sorting) pairs.  Groups
    without a readable .res/.clu are omitted from sortings but kept in
    recordings."""
    sess = load_session_yaml(session_yaml)
    recordings, sortings = {}, {}
    for g_idx, chans in enumerate(sess.spike_groups, start=1):
        recordings[g_idx] = NS3RecordingExtractor(
            session_yaml, channels=chans, raw=raw,
            transform=transform, name=f"ns3-g{g_idx}",
        )
        try:
            sortings[g_idx] = NS3SortingExtractor(
                session_yaml, electrode_group=g_idx,
                variant=variant, tag=tag, exclude_reserved=exclude_reserved,
                name=f"ns3-g{g_idx}-sort",
            )
        except FileNotFoundError:
            continue
    return recordings, sortings
