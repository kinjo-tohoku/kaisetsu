"""merge_ranges / HistogramEntry.set_excluded の正規化。"""

from __future__ import annotations

from kaisetsu.refine.project import merge_ranges


def test_merge_ranges_sorts_merges_and_drops_empty() -> None:
    assert merge_ranges([]) == []
    assert merge_ranges([(5.0, 3.0)]) == [(3.0, 5.0)]
    assert merge_ranges([(1.0, 1.0)]) == []
    assert merge_ranges([(10.0, 12.0), (1.0, 2.0), (11.5, 13.0), (13.0, 14.0)]) == [
        (1.0, 2.0),
        (10.0, 14.0),
    ]
    assert merge_ranges([(2, 3), (1, 10)]) == [(1.0, 10.0)]
