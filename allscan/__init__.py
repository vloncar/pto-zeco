"""AllScan collective: the communication primitive at the core of ZeCO.

Exposes the :class:`~allscan.common.AllscanImpl` interface, the sequential
reference math, and (via :mod:`allscan.implementations`) the torch / simpler /
pypto backends. The GLA operator in :mod:`gla` consumes these for its
cross-device state hand-off.
"""
