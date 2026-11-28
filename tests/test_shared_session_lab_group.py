"""A lab electrode group's shared session holds only that group's channels."""
import os

import numpy as np
import pytest

si = pytest.importorskip("spikeinterface.core")

from ns3_sort_bench._base import Corpus  # noqa: E402
from ns3_sort_bench.klusters_export import prepare_shared_session  # noqa: E402


def test_lab_group_gets_its_own_fil(tmp_path):
    n, n_all = 5000, 16
    sess = tmp_path / "lab" / "sess"
    sess.parent.mkdir()
    full = (np.arange(n * n_all, dtype=np.int64) % 30000 - 15000) \
        .astype(np.int16).reshape(n, n_all)
    full.tofile(str(sess) + ".fil")
    (tmp_path / "lab" / "sess.yaml").write_text("acquisitionSystem: {}\n")
    group = [8, 9, 10, 11, 12, 13, 14, 15]
    rec = si.NumpyRecording([full[:, group]], 20000.0)
    rec.set_dummy_probe_from_locations(
        np.column_stack([np.zeros(8), 20.0 * np.arange(8)]))
    rec.annotate(session_base=str(sess))
    corpus = Corpus(name="lab_g8", recording=rec, ground_truth=None)

    base = prepare_shared_session(str(tmp_path / "out"), corpus)
    got = np.fromfile(base + ".fil", np.int16).reshape(-1, 8)
    assert np.array_equal(got, full[:, group])     # copied, not re-filtered
    assert os.path.samefile(base + ".fil", base + ".dat")
    assert not os.path.samefile(base + ".fil", str(sess) + ".fil")
    assert "nChannels: 8" in open(base + ".yaml").read().replace("'", "")
    assert os.path.exists(base + ".probe")
    # idempotent
    assert prepare_shared_session(str(tmp_path / "out"), corpus) == base


def test_whole_session_recording_still_links(tmp_path):
    n = 4000
    sess = tmp_path / "sess"
    data = np.ones((n, 4), np.int16)
    data.tofile(str(sess) + ".fil")
    rec = si.NumpyRecording([data], 20000.0)
    rec.annotate(session_base=str(sess))
    base = prepare_shared_session(str(tmp_path / "out"),
                                  Corpus(name="whole", recording=rec,
                                         ground_truth=None))
    assert os.path.samefile(base + ".fil", str(sess) + ".fil")
