"""hnl.trackml.events.table_root: each event is read from the tables of its phase (ids do not overlap)."""

from hnl.trackml import events as E
from hnl.trackml.convert import DERIVED


def test_table_root_by_phase():
    assert E.table_root(1000) == DERIVED and E.table_root(9999) == DERIVED          # accuracy: smoke, dev, T2
    assert E.table_root(21000) == E.DERIVED_TP and E.table_root(29999) == E.DERIVED_TP   # throughput parts
    assert E.DERIVED_TP.endswith("derived/tp_v1") and DERIVED.endswith("derived/v1")
