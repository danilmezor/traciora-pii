"""OPF detector behaviour that does not need the model weights."""
import os
import sys
import types

import pytest

from traciora_pii.opf_pii_detector import OPFInferenceError, OPFPIIDetector


def detector_with(opf, device="cpu"):
    detector = OPFPIIDetector.__new__(OPFPIIDetector)
    detector.device = device
    detector.prefilter_enabled = False
    detector._label_to_placeholder = {"private_person": "[PERSON_NAME]"}
    detector.lm_calls = detector.prefilter_skips = 0
    detector.hallucinated_drops = detector.parse_failures = 0
    detector._opf = opf
    return detector


def test_detect_raises_when_inference_fails():
    class Broken:
        def redact(self, text):
            raise RuntimeError("triton")

    detector = detector_with(Broken(), device="mps")
    with pytest.raises(OPFInferenceError) as excinfo:
        detector.detect('{"note": "call Maria Schneider"}')
    assert "triton" in str(excinfo.value)
    assert "device=mps" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)
    assert detector.lm_calls == 1


class RecordingOPF:
    seen: list = []

    def __init__(self, **kwargs):
        RecordingOPF.seen.append(os.environ.get("OPF_MOE_TRITON"))


@pytest.fixture
def fake_opf(monkeypatch):
    module = types.ModuleType("opf")
    module.OPF = RecordingOPF
    monkeypatch.setitem(sys.modules, "opf", module)
    monkeypatch.setattr(RecordingOPF, "seen", [])
    return RecordingOPF


def test_mps_sets_torch_ops_moe_when_unset(fake_opf, monkeypatch):
    monkeypatch.delenv("OPF_MOE_TRITON", raising=False)
    OPFPIIDetector(device="mps", label_map={})
    assert fake_opf.seen == ["0"]


def test_mps_respects_explicit_triton_setting(fake_opf, monkeypatch):
    monkeypatch.setenv("OPF_MOE_TRITON", "1")
    OPFPIIDetector(device="mps", label_map={})
    assert fake_opf.seen == ["1"]


def test_cpu_leaves_moe_setting_alone(fake_opf, monkeypatch):
    monkeypatch.delenv("OPF_MOE_TRITON", raising=False)
    OPFPIIDetector(device="cpu", label_map={})
    assert fake_opf.seen == [None]
