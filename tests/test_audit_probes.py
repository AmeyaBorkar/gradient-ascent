"""Row groups from the device counter, on hand-built arrays."""

from __future__ import annotations

import numpy as np

from ringtrace.audit import probes
from ringtrace.tables import HOME_DEVICES


def test_evidence_reads_events_and_marks():
    ev1 = np.array([np.nan, 3.0, -1.0, np.nan])
    ev2 = np.array([np.nan, np.nan, np.nan, 0.0])
    marks = np.array([0, 0, 1, np.nan])
    assert probes.evidence([ev1, ev2], marks).tolist() == [False, True, True, True]


def test_device_kind():
    h = HOME_DEVICES
    device = np.array([5, 7, h + 10, h + 20, h + 30, h + 40])
    sender = np.array([5, 5, 1, 1, 1, 1])
    block = np.array([0, 0, 1, 2, 3, 0])
    kind = probes.device_kind(device, sender, block)
    assert [probes.KIND_NAMES[k] for k in kind] == [
        "own",
        "borrowed",
        "bot",
        "fraud_phase",
        "legit_block",
        "unplaced",
    ]


def test_ordinary_probe_needs_single_phone_and_no_evidence():
    k = probes
    kind = np.array(
        [k.FRAUD_PHASE, k.FRAUD_PHASE, k.FRAUD_PHASE, k.LEGIT_BLOCK, k.LEGIT_BLOCK, k.OWN]
    )
    single = np.array([1, 0, 1, 1, 1, 1])
    evid = np.array([False, False, True, False, True, False])
    assert probes.ordinary_probe(kind, single, evid).tolist() == [1, -1, -1, 0, -1, -1]


def test_segment_class_shifts_and_blanks():
    assert probes.segment_class(np.array([0.0, 10.0, np.nan])).tolist() == [1, 11, 0]
