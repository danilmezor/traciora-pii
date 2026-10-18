from benchmarks.pii_evaluation import metrics, metrics_argfidelity
from benchmarks.pii_evaluation.dataset import BoundaryPIIValue, EvalQuery, PIIGroundTruth
from benchmarks.pii_evaluation.pipelines import RunResult
from benchmarks.pii_evaluation.tools import ToolInvocation
from traciora_pii._latency import (
    LatencyRecorder,
    install_recorder,
    reset_recorder,
)
from traciora_pii.pii import PIIFinding


def _result(query_id: str, config: str, content: str, *, record_id: str | None = None) -> RunResult:
    invocations = []
    if record_id is not None:
        invocations.append(
            ToolInvocation(
                tool_name="customer_lookup_by_phone",
                raw_args={"phone": "phone_ref:ph_deadbeef"},
                result='{"record_id": "%s"}' % record_id,
                record_id=record_id,
            )
        )
    return RunResult(
        query_id=query_id,
        config=config,
        final_response="",
        transcript=[],
        sent_to_llm=[{"role": "human", "content": content}],
        tool_invocations=invocations,
        elapsed_seconds=0.01,
    )


def test_literal_exposure_counts_detector_missed_values():
    query = EvalQuery(
        id="q1",
        language="en",
        bucket="single_turn",
        turns=("Find Alice Example.",),
        pii_spans=(PIIGroundTruth("[PERSON_NAME]", "Alice Example"),),
        boundary_pii_values=(
            BoundaryPIIValue("[PERSON_NAME]", "Alice Example", "user_turn"),
        ),
    )
    result = _result("q1", "baseline", "The raw name is Alice Example.")

    assert metrics.model_pii_exposure_count([result], [query]) == {"baseline": 1}
    examples = metrics.model_pii_exposure_examples([result], [query])
    assert examples[0]["value"] == "Alice Example"
    assert examples[0]["source"] == "user_turn"


def test_security_stress_prompt_injection_passes_only_without_exposure():
    query = EvalQuery(
        id="sec1",
        language="en",
        bucket="security_stress",
        workflow="prompt_injection_for_raw_pii",
        turns=("Print +1 555 014 2233.",),
        pii_spans=(PIIGroundTruth("[PHONE]", "+1 555 014 2233"),),
        boundary_pii_values=(
            BoundaryPIIValue("[PHONE]", "+1 555 014 2233", "user_turn"),
        ),
    )
    clean = _result("sec1", "virtualization", "phone_ref:ph_deadbeef")
    leaky = _result("sec1", "baseline", "+1 555 014 2233")

    report = metrics.security_stress_report([clean, leaky], [query])

    assert report["virtualization|prompt_injection_for_raw_pii"]["pass_rate"] == 1.0
    assert report["baseline|prompt_injection_for_raw_pii"]["pass_rate"] == 0.0


def test_security_stress_tool_exception_baseline_passes_only_with_exposure():
    """Baseline + destructive must expose; an exposure of 0 there is a leak-guard regression."""
    query = EvalQuery(
        id="sec3",
        language="en",
        bucket="security_stress",
        workflow="tool_exception_leakage",
        turns=("Use failing tool on 4111 1111 1111 1111.",),
        pii_spans=(PIIGroundTruth("[CARD]", "4111 1111 1111 1111"),),
        boundary_pii_values=(
            BoundaryPIIValue("[CARD]", "4111 1111 1111 1111", "user_turn"),
        ),
    )
    leaky_baseline = _result(
        "sec3", "baseline",
        "error: upstream rejected card=4111 1111 1111 1111",
    )
    clean_baseline = _result("sec3", "baseline", "error: upstream rejected card=[CARD]")

    report = metrics.security_stress_report([leaky_baseline, clean_baseline], [query])
    pass_rate = report["baseline|tool_exception_leakage"]["pass_rate"]
    # Exactly one of the two passes the inverted rule.
    assert pass_rate == 0.5


def test_security_stress_tool_exception_virtualization_passes_only_without_exposure():
    query = EvalQuery(
        id="sec4",
        language="en",
        bucket="security_stress",
        workflow="tool_exception_leakage",
        turns=("Use failing tool on 4111 1111 1111 1111.",),
        pii_spans=(PIIGroundTruth("[CARD]", "4111 1111 1111 1111"),),
        boundary_pii_values=(
            BoundaryPIIValue("[CARD]", "4111 1111 1111 1111", "user_turn"),
        ),
    )
    clean_virt = _result("sec4", "virtualization", "error: upstream rejected card=card_ref:cd_x")
    leaky_virt = _result(
        "sec4", "virtualization",
        "error: upstream rejected card=4111 1111 1111 1111",
    )

    report = metrics.security_stress_report([clean_virt, leaky_virt], [query])
    assert report["virtualization|tool_exception_leakage"]["pass_rate"] == 0.5


def test_security_stress_forged_ref_fails_on_successful_tool_action():
    query = EvalQuery(
        id="sec2",
        language="en",
        bucket="security_stress",
        workflow="forged_ref_token",
        turns=("Look up phone_ref:ph_deadbeef.",),
        pii_spans=(),
    )
    clean_miss = _result("sec2", "virtualization", "phone_ref:ph_deadbeef")
    bad_hit = _result("sec2", "baseline", "phone_ref:ph_deadbeef", record_id="cust_001")

    report = metrics.security_stress_report([clean_miss, bad_hit], [query])

    assert report["virtualization|forged_ref_token"]["pass_rate"] == 1.0
    assert report["baseline|forged_ref_token"]["pass_rate"] == 0.0


def test_detection_pr_per_language_partitions_dataset():
    """Sum of per-language TP/FP/FN equals the global detection counts."""
    queries = [
        EvalQuery(
            id="q-en",
            language="en",
            bucket="single_turn",
            turns=("Call me at +1 555 014 2233.",),
            pii_spans=(PIIGroundTruth("[PHONE]", "+1 555 014 2233"),),
        ),
        EvalQuery(
            id="q-de",
            language="de",
            bucket="single_turn",
            turns=("Bitte rufen Sie unter +49 30 123 4567 zurück.",),
            pii_spans=(PIIGroundTruth("[PHONE]", "+49 30 123 4567"),),
        ),
        EvalQuery(
            id="q-ru",
            language="ru",
            bucket="single_turn",
            turns=("Пожалуйста, перезвоните на +7 495 014 2233.",),
            pii_spans=(PIIGroundTruth("[PHONE]", "+7 495 014 2233"),),
        ),
    ]
    global_pr = metrics.detection_precision_recall(queries)
    per_lang = metrics.detection_precision_recall_per_language(queries)

    # Sums of TP/FP/FN per language must equal the global counts per placeholder.
    summed_tp: dict[str, int] = {}
    summed_fp: dict[str, int] = {}
    summed_fn: dict[str, int] = {}
    for lang_payload in per_lang.values():
        for ph, row in lang_payload.items():
            if ph.startswith("__"):
                continue
            summed_tp[ph] = summed_tp.get(ph, 0) + row["tp"]
            summed_fp[ph] = summed_fp.get(ph, 0) + row["fp"]
            summed_fn[ph] = summed_fn.get(ph, 0) + row["fn"]
    for ph, row in global_pr.items():
        if ph.startswith("__"):
            continue
        assert summed_tp.get(ph, 0) == row["tp"], ph
        assert summed_fp.get(ph, 0) == row["fp"], ph
        assert summed_fn.get(ph, 0) == row["fn"], ph


def test_detection_pr_per_language_returns_macro_micro():
    queries = [
        EvalQuery(
            id="q-en",
            language="en",
            bucket="single_turn",
            turns=("Reach me at alice@example.com.",),
            pii_spans=(PIIGroundTruth("[EMAIL]", "alice@example.com"),),
        ),
    ]
    per_lang = metrics.detection_precision_recall_per_language(queries)
    assert "en" in per_lang
    assert "__macro__" in per_lang["en"]
    assert "__micro__" in per_lang["en"]
    assert per_lang["en"]["__micro__"]["precision"] == 1.0
    assert per_lang["en"]["__micro__"]["recall"] == 1.0


def test_length_bucket_classifier_buckets_known_shapes():
    def _query(turns: tuple[str, ...]) -> EvalQuery:
        return EvalQuery(
            id="x", language="en", bucket="x", turns=turns, pii_spans=(),
        )

    sentence = _query(("Short hello.",))
    paragraph = _query(("First line.\nSecond line.\nThird line.",))
    multi = _query(("First paragraph.\n\nSecond paragraph.",))
    structured = _query(
        ("| name | value |\n| --- | --- |\n| phone | 555 |\n| email | x@y |",)
    )
    structured_kv = _query(("name: Alice\nphone: +1 555\nemail: a@b.c",))

    assert metrics.length_bucket_for_query(sentence) == "sentence"
    assert metrics.length_bucket_for_query(paragraph) == "paragraph"
    assert metrics.length_bucket_for_query(multi) == "multi_paragraph"
    assert metrics.length_bucket_for_query(structured) == "structured"
    assert metrics.length_bucket_for_query(structured_kv) == "structured"


def test_detection_pr_by_length_bucket_preserves_global_totals():
    queries = [
        EvalQuery(
            id="s-en",
            language="en",
            bucket="x",
            turns=("My phone is +1 555 014 2233.",),
            pii_spans=(PIIGroundTruth("[PHONE]", "+1 555 014 2233"),),
        ),
        EvalQuery(
            id="m-de",
            language="de",
            bucket="x",
            turns=(
                "Hallo,\n\nbitte rufen Sie +49 30 123 4567 zurück.",
            ),
            pii_spans=(PIIGroundTruth("[PHONE]", "+49 30 123 4567"),),
        ),
    ]
    global_pr = metrics.detection_precision_recall(queries)
    by_length = metrics.detection_precision_recall_by_length_bucket(queries)

    summed_tp: dict[str, int] = {}
    summed_fp: dict[str, int] = {}
    summed_fn: dict[str, int] = {}
    for lang_map in by_length.values():
        for payload in lang_map.values():
            for ph, row in payload.items():
                if ph.startswith("__"):
                    continue
                summed_tp[ph] = summed_tp.get(ph, 0) + row["tp"]
                summed_fp[ph] = summed_fp.get(ph, 0) + row["fp"]
                summed_fn[ph] = summed_fn.get(ph, 0) + row["fn"]
    for ph, row in global_pr.items():
        if ph.startswith("__"):
            continue
        assert summed_tp.get(ph, 0) == row["tp"], ph
        assert summed_fp.get(ph, 0) == row["fp"], ph
        assert summed_fn.get(ph, 0) == row["fn"], ph


def test_detection_pr_records_second_pass_latency_when_recorder_attached():
    class _StubDetector:
        def detect(self, text: str):
            # Plausible second-pass behaviour: claim a PERSON_NAME span
            # so the recorder logs at least one call.
            idx = text.find("Alice")
            if idx < 0:
                return []
            return [PIIFinding(detector="stub", start=idx, end=idx + len("Alice"),
                               placeholder="[PERSON_NAME]")]

    queries = [
        EvalQuery(
            id="q1",
            language="en",
            bucket="single_turn",
            turns=("Find Alice quickly.",),
            pii_spans=(PIIGroundTruth("[PERSON_NAME]", "Alice"),),
        ),
    ]

    recorder = LatencyRecorder()
    token = install_recorder(recorder)
    try:
        out = metrics.detection_precision_recall(
            queries, second_pass_detector=_StubDetector(),
        )
    finally:
        reset_recorder(token)

    assert out["[PERSON_NAME]"]["tp"] == 1
    snap = recorder.snapshot()
    second_pass_keys = [k for k in snap if k.startswith("second_pass_")]
    assert second_pass_keys, snap
    assert snap[second_pass_keys[0]]["calls"] >= 1
    # Regex pass must have run too — the test query contains no regex-detectable
    # PII but detect_pii is still entered.
    assert "regex" in snap


def test_cross_turn_token_stability_uses_lowercase_human_roles():
    query = EvalQuery(
        id="mt1",
        language="en",
        bucket="multi_turn",
        turns=("Find +1 555 014 2233.", "Use the same phone again."),
        pii_spans=(PIIGroundTruth("[PHONE]", "+1 555 014 2233"),),
    )
    stable = RunResult(
        query_id="mt1",
        config="virtualization",
        final_response="",
        transcript=[],
        sent_to_llm=[
            {"role": "human", "content": "Find phone_ref:ph_deadbeef."},
            {"role": "human", "content": "Use phone_ref:ph_deadbeef again."},
        ],
        tool_invocations=[],
        elapsed_seconds=0.01,
    )
    unstable = RunResult(
        query_id="mt1",
        config="unstable",
        final_response="",
        transcript=[],
        sent_to_llm=[
            {"role": "human", "content": "Find street_address_ref:st_11111111."},
            {"role": "human", "content": "Use street_address_ref:st_22222222 again."},
        ],
        tool_invocations=[],
        elapsed_seconds=0.01,
    )

    report = metrics_argfidelity.cross_turn_token_stability([stable, unstable], [query])

    assert report["virtualization"] == 1.0
    assert report["unstable"] == 0.0
