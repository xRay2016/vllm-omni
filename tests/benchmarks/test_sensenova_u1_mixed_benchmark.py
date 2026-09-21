# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from benchmarks.sensenova_u1.benchmark_mixed_serving import (
    BenchmarkCase,
    RequestResult,
    load_cases,
    make_payload,
    percentile_metrics,
    summarize,
)

pytestmark = [pytest.mark.core_model, pytest.mark.benchmark, pytest.mark.cpu]


def test_default_sequence_covers_mixed_traffic_and_shape_reuse():
    cases = load_cases(None, steps=8, cfg_scale=4.0, max_tokens=64)

    assert {case.modality for case in cases} == {"text2img", "img2img", "text2text", "img2text"}
    assert any(case.think for case in cases)
    assert [case.group for case in cases].count("t2i_1024_no_think") == 2
    assert [case.group for case in cases].count("t2t") == 2


def test_make_payload_routes_image_and_generation_parameters():
    case = BenchmarkCase(
        name="edit",
        group="edit",
        modality="img2img",
        prompt="paint it",
        width=1024,
        height=768,
        think=True,
        uses_image=True,
        num_inference_steps=8,
        cfg_scale=4.0,
    )

    payload = make_payload(case, model="model", image_data_url="data:image/png;base64,AAAA", seed=42)

    assert payload["modalities"] == ["image"]
    assert payload["width"] == 1024
    assert payload["height"] == 768
    assert payload["num_inference_steps"] == 8
    assert payload["think"] is True
    assert payload["messages"][0]["content"][1]["type"] == "image_url"


def test_required_image_is_rejected_without_input():
    case = BenchmarkCase(name="understand", group="understand", modality="img2text", prompt="describe", uses_image=True)

    with pytest.raises(ValueError, match="requires an input image"):
        make_payload(case, model="model", image_data_url=None, seed=42)


def test_summary_reports_p50_p100_and_groups():
    results = [
        RequestResult("cold", "shape", "text2img", 0, 0, 4.0, True, 200, 1),
        RequestResult("repeat", "shape", "text2img", 0, 1, 2.0, True, 200, 1),
        RequestResult("failure", "other", "text2text", 0, 2, 1.0, False, 500, 0, "HTTP 500"),
    ]

    summary = summarize(results)

    assert summary["successful"] == 2
    assert summary["failed"] == 1
    assert summary["latency"] == {"count": 2, "mean_s": 3.0, "p50_s": 3.0, "p100_s": 4.0}
    assert summary["latency_by_group"]["shape"]["p100_s"] == 4.0
    assert percentile_metrics([])["count"] == 0
