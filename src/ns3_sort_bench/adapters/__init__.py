"""Sorter adapters.  Each self-registers in `adapter_registry` on import.
Dependencies absent → disabled-with-reason stub."""
from . import botm_ref  # noqa: F401
from . import si_sorters  # noqa: F401  (MS5, TDC2, SC2, KS4)
from . import dartsort  # noqa: F401
from . import slay  # noqa: F401

# fiber_kit is optional; try/except so a bare install doesn't error at
# import time just because `fiber_kit` is missing from the env.
try:
    from . import fiber_kit  # noqa: F401
except ImportError:
    pass

# Lattice (32 fiber_deconv ablation configs) also needs fiber_kit.
try:
    from . import lattice_botm  # noqa: F401
except ImportError:
    pass
