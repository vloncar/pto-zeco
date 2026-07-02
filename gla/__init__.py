"""ZeCO: sequence-parallel Gated Linear Attention built on the AllScan collective.

Exposes the :class:`~gla.common.ZeCoImpl` interface, the sequential
:func:`~gla.common.expected_gla` golden, the shared chunk-wise GLA building
blocks, and (via :mod:`gla.implementations`) the backends. Each backend performs
the local GLA compute and defers the cross-device boundary-state hand-off to an
:mod:`allscan` backend.
"""
