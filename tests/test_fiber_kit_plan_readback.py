"""fiber_kit reads back the stages of the plan that actually ran."""
import os

from ns3_sort_bench.adapters.fiber_kit import (
    _FIBER_KIT_SCOREABLE_STAGES, _FiberKitBase, plan_outputs,
)
from ns3_sort_bench.configs import load_adapter_config_path

HERE = os.path.dirname(__file__)


def test_user_plan_t2():
    final, earlier = plan_outputs(os.path.join(HERE, "data", "fk_plan_t2_0312.yaml"))
    assert final == "fc_unmix_t2"
    assert earlier == ["fc_session", "fc_polish_t2", "fc_clean_t2", "fc_xcm_t2"]


def test_bundled_plan_matches_the_fixed_tags():
    final, earlier = plan_outputs(load_adapter_config_path("fiber_kit"))
    assert final == _FiberKitBase._MODE_FINAL_TAG["fiber_kit"]
    assert tuple(earlier) == _FIBER_KIT_SCOREABLE_STAGES


def test_no_plan(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("FK_SESSION_JOBS: 0\n")
    assert plan_outputs(str(p)) is None
    p.write_text("pipeline: \n")
    assert plan_outputs(str(p)) is None
    assert plan_outputs(str(tmp_path / "missing.yaml")) is None
