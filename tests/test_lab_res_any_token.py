"""A lab session's shared .res is found whatever token it was detected
under: the ground truth of lab:<yaml>:<g>:stderiv_C5:cur_pl reads over a
.res.stderiv_S3.<g> (the curated stage has no .res of its own)."""
import numpy as np
import pytest
import yaml

from ns3_sort_bench.ns3 import extractors as ex


def _session(d, res_tokens):
    y = {"acquisitionSystem": {"samplingRate": 20000.0, "nChannels": 2, "nBits": 16},
         "spikeDetection": {"channelGroups": [{"channels": [0, 1], "nSamples": 4,
                                               "peakSampleIndex": 2, "nFeatures": 3}]}}
    b = str(d / "rat")
    with open(b + ".yaml", "w") as f:
        yaml.safe_dump(y, f)
    for t in res_tokens:
        np.array([10, 20, 30, 40], "<i8").tofile(f"{b}.res.{t}.1")
    np.array([3, 2, 2, 3, 3], "<i4").tofile(f"{b}.clu.stderiv_C5.1.cur_pl")
    return b


@pytest.mark.parametrize("have_fk", [True, False])
def test_shared_res_under_another_token(tmp_path, monkeypatch, have_fk):
    if not have_fk:
        monkeypatch.setattr(ex, "HAVE_FK", False)
    elif not ex.HAVE_FK:
        pytest.skip("fiber-kit not installed")
    b = _session(tmp_path, ["stderiv_S3"])
    res, clu = ex._resolve_sort_pair(b, 1, variant="stderiv_C5", tag="cur_pl")
    assert res.endswith("rat.res.stderiv_S3.1") and clu.endswith("rat.clu.stderiv_C5.1.cur_pl")
    s = ex.NS3SortingExtractor(b + ".yaml", electrode_group=1, variant="stderiv_C5", tag="cur_pl")
    assert sorted(map(int, s.get_unit_ids())) == [2, 3]
    assert s.get_unit_spike_train(2).tolist() == [10, 20]


def test_two_unrelated_tokens_are_not_guessed_without_fiber_kit(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "HAVE_FK", False)
    b = _session(tmp_path, ["stderiv_S3", "stderiv_S4"])
    with pytest.raises(FileNotFoundError, match="any"):
        ex._resolve_sort_pair(b, 1, variant="stderiv_C5", tag="cur_pl")
