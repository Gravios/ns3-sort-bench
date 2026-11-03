"""Sibling-implementation lattice variants (pursue_variant with configurable
overlap-resolution rule).  Imports fiber_kit primitives — only usable when
fiber_kit is installed (`pip install ns3-sort-bench[fiber-kit]`)."""
from .fiber_deconv_lattice import pursue_variant, PURSUE_RULES

__all__ = ["pursue_variant", "PURSUE_RULES"]
