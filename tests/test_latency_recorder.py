"""Tests for the per-detector latency recorder."""
from __future__ import annotations

import time

from traciora_pii._latency import (
    LatencyRecorder,
    get_active_recorder,
    install_recorder,
    reset_recorder,
    time_detector,
)


def test_time_detector_is_no_op_when_unattached():
    """Without an installed recorder, time_detector raises nothing and records nothing."""
    assert get_active_recorder() is None
    with time_detector("regex"):
        x = sum(range(10))
    assert x == 45  # block executed
    assert get_active_recorder() is None


def test_recorder_aggregates_calls():
    recorder = LatencyRecorder()
    token = install_recorder(recorder)
    try:
        with time_detector("regex"):
            time.sleep(0.001)
        with time_detector("regex"):
            time.sleep(0.001)
        with time_detector("second_pass_OPFPIIDetector"):
            time.sleep(0.002)
    finally:
        reset_recorder(token)

    snap = recorder.snapshot()
    assert snap["regex"]["calls"] == 2
    assert snap["regex"]["total_s"] > 0
    assert snap["second_pass_OPFPIIDetector"]["calls"] == 1
    assert snap["second_pass_OPFPIIDetector"]["mean_ms"] >= 1.0


def test_reset_recorder_detaches():
    recorder = LatencyRecorder()
    token = install_recorder(recorder)
    reset_recorder(token)
    assert get_active_recorder() is None
    with time_detector("regex"):
        pass
    assert recorder.snapshot() == {}


def test_recorder_reset_clears_samples():
    recorder = LatencyRecorder()
    token = install_recorder(recorder)
    try:
        with time_detector("regex"):
            pass
    finally:
        reset_recorder(token)
    assert recorder.snapshot() != {}
    recorder.reset()
    assert recorder.snapshot() == {}
