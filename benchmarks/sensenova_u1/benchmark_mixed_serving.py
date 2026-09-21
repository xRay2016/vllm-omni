#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sequential mixed-traffic benchmark for SenseNova-U1 and U1.5 serving.

The benchmark intentionally defaults to no client-side warmup.  Its first
cycle therefore exposes work deferred until after engine readiness, while
later cycles show whether the same shapes are reused without another latency
spike.  Start a fresh server for every configuration being compared.
"""

from __future__ import annotations

import argparse
import base64
import json
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

DEFAULT_SEQUENCE = (
    {"name": "t2i_1024_no_think", "modality": "text2img", "width": 1024, "height": 1024},
    {"name": "t2i_1024_think", "modality": "text2img", "width": 1024, "height": 1024, "think": True},
    {"name": "t2t", "modality": "text2text"},
    {"name": "i2t", "modality": "img2text", "uses_image": True},
    {"name": "i2i_1024_no_think", "modality": "img2img", "width": 1024, "height": 1024, "uses_image": True},
    {"name": "t2i_1536_no_think", "modality": "text2img", "width": 1536, "height": 1536},
    # Deliberate repeats: these distinguish first-seen shape cost from reuse.
    {"name": "t2i_1024_repeat", "group": "t2i_1024_no_think", "modality": "text2img", "width": 1024, "height": 1024},
    {"name": "t2t_repeat", "group": "t2t", "modality": "text2text"},
)


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    modality: str
    prompt: str
    group: str | None = None
    width: int | None = None
    height: int | None = None
    think: bool = False
    uses_image: bool = False
    num_inference_steps: int | None = None
    cfg_scale: float | None = None
    max_tokens: int | None = None


@dataclass
class RequestResult:
    name: str
    group: str
    modality: str
    cycle: int
    sequence_index: int
    latency_s: float
    success: bool
    status_code: int | None
    response_bytes: int
    error: str | None = None


def _default_prompt(modality: str) -> str:
    if modality == "text2text":
        return "What is the capital of France? Answer briefly."
    if modality == "img2text":
        return "Describe this image briefly."
    if modality == "img2img":
        return "Turn this image into a watercolor painting."
    return "A red apple on a wooden table, studio lighting."


def load_cases(path: str | None, *, steps: int, cfg_scale: float, max_tokens: int) -> list[BenchmarkCase]:
    if path is None:
        raw_cases: Any = DEFAULT_SEQUENCE
    else:
        with open(path, encoding="utf-8") as file:
            raw_cases = json.load(file)
    if not isinstance(raw_cases, list) and not isinstance(raw_cases, tuple):
        raise ValueError("benchmark sequence must be a JSON list")

    cases: list[BenchmarkCase] = []
    valid_modalities = {"text2img", "img2img", "img2text", "text2text"}
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"sequence entry {index} must be an object")
        modality = str(raw.get("modality", ""))
        if modality not in valid_modalities:
            raise ValueError(f"sequence entry {index} has unsupported modality {modality!r}")
        name = str(raw.get("name") or f"{modality}_{index}")
        is_image_output = modality in {"text2img", "img2img"}
        uses_image = bool(raw.get("uses_image", modality in {"img2img", "img2text"}))
        cases.append(
            BenchmarkCase(
                name=name,
                group=str(raw.get("group") or name),
                modality=modality,
                prompt=str(raw.get("prompt") or _default_prompt(modality)),
                width=int(raw["width"]) if raw.get("width") is not None else None,
                height=int(raw["height"]) if raw.get("height") is not None else None,
                think=bool(raw.get("think", False)),
                uses_image=uses_image,
                num_inference_steps=(int(raw.get("num_inference_steps", steps)) if is_image_output else None),
                cfg_scale=(float(raw.get("cfg_scale", cfg_scale)) if is_image_output else None),
                max_tokens=(int(raw.get("max_tokens", max_tokens)) if not is_image_output else None),
            )
        )
    if not cases:
        raise ValueError("benchmark sequence must contain at least one request")
    return cases


def encode_image(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def make_payload(case: BenchmarkCase, *, model: str, image_data_url: str | None, seed: int) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": case.prompt}]
    if case.uses_image:
        if image_data_url is None:
            raise ValueError(f"case {case.name!r} requires an input image")
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "modalities": ["image" if case.modality in {"text2img", "img2img"} else "text"],
        "seed": seed,
    }
    if case.width is not None:
        payload["width"] = case.width
    if case.height is not None:
        payload["height"] = case.height
    if case.num_inference_steps is not None:
        payload["num_inference_steps"] = case.num_inference_steps
    if case.cfg_scale is not None:
        payload["cfg_scale"] = case.cfg_scale
    if case.max_tokens is not None:
        payload["max_tokens"] = case.max_tokens
    if case.think:
        payload["think"] = True
    return payload


def percentile_metrics(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean_s": 0.0, "p50_s": 0.0, "p100_s": 0.0}
    return {
        "count": len(values),
        "mean_s": float(np.mean(values)),
        "p50_s": float(np.percentile(values, 50)),
        "p100_s": float(max(values)),
    }


def summarize(results: list[RequestResult]) -> dict[str, Any]:
    successful = [result for result in results if result.success]
    grouped: dict[str, list[float]] = {}
    by_cycle: dict[str, list[float]] = {}
    for result in successful:
        grouped.setdefault(result.group, []).append(result.latency_s)
        by_cycle.setdefault(str(result.cycle), []).append(result.latency_s)
    return {
        "requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "latency": percentile_metrics([result.latency_s for result in successful]),
        "latency_by_group": {name: percentile_metrics(values) for name, values in grouped.items()},
        "latency_by_cycle": {cycle: percentile_metrics(values) for cycle, values in by_cycle.items()},
    }


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def environment_metadata() -> dict[str, Any]:
    return {
        "commit_sha": _command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(_command_output(["git", "status", "--porcelain"])),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "gpu": _command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }


def _response_is_valid(response: requests.Response, case: BenchmarkCase) -> bool:
    try:
        body = response.json()
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError):
        return False
    if case.modality in {"text2img", "img2img"}:
        return isinstance(content, list) and bool(content) and "image_url" in content[0]
    return isinstance(content, str) and bool(content)


def run(args: argparse.Namespace) -> dict[str, Any]:
    cases = load_cases(
        args.sequence_file, steps=args.num_inference_steps, cfg_scale=args.cfg_scale, max_tokens=args.max_tokens
    )
    api_url = f"{args.server.rstrip('/')}/v1/chat/completions"

    if args.input_image:
        image_path = Path(args.input_image)
        if not image_path.is_file():
            raise ValueError(f"input image does not exist: {image_path}")
    else:
        image_path = Path(tempfile.gettempdir()) / "sensenova_mixed_benchmark_input.png"
        Image.new("RGB", (args.input_image_size, args.input_image_size), (90, 140, 190)).save(image_path)
    image_data_url = encode_image(image_path)

    results: list[RequestResult] = []
    with requests.Session() as session:
        for cycle in range(args.cycles):
            for sequence_index, case in enumerate(cases):
                payload = make_payload(case, model=args.model, image_data_url=image_data_url, seed=args.seed)
                start = time.perf_counter()
                status_code: int | None = None
                error: str | None = None
                response_bytes = 0
                success = False
                try:
                    response = session.post(api_url, json=payload, timeout=args.timeout)
                    status_code = response.status_code
                    response_bytes = len(response.content)
                    success = response.ok and _response_is_valid(response, case)
                    if not success:
                        error = f"HTTP {response.status_code}" if not response.ok else "unexpected response schema"
                except requests.RequestException as exc:
                    error = str(exc)
                latency = time.perf_counter() - start
                result = RequestResult(
                    name=case.name,
                    group=case.group or case.name,
                    modality=case.modality,
                    cycle=cycle,
                    sequence_index=sequence_index,
                    latency_s=latency,
                    success=success,
                    status_code=status_code,
                    response_bytes=response_bytes,
                    error=error,
                )
                results.append(result)
                print(
                    f"cycle={cycle} index={sequence_index} case={case.name} "
                    f"latency={latency:.3f}s status={'ok' if success else error}",
                    flush=True,
                )
                if not success and args.fail_fast:
                    raise RuntimeError(f"benchmark request {case.name!r} failed: {error}")

    report = {
        "schema_version": 1,
        "benchmark": "sensenova_u1_mixed_serving",
        "created_at_unix": time.time(),
        "environment": environment_metadata(),
        "configuration": {
            "server": args.server,
            "model": args.model,
            "cycles": args.cycles,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "max_tokens": args.max_tokens,
            "input_image": str(image_path),
            "server_configuration": args.server_configuration,
            "client_warmup_requests": 0,
        },
        "sequence": [asdict(case) for case in cases],
        "summary": summarize(results),
        "requests": [asdict(result) for result in results],
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark sequential mixed traffic against a SenseNova-U1 server.")
    parser.add_argument("--server", default="http://localhost:8091")
    parser.add_argument("--model", default="sensenova/SenseNova-U1.5-8B-MoT")
    parser.add_argument("--sequence-file", help="JSON request sequence; uses the built-in mixed sequence by default")
    parser.add_argument("--input-image", help="Local image used by I2I and I2T cases")
    parser.add_argument("--input-image-size", type=int, default=1024)
    parser.add_argument("--cycles", type=int, default=1, help="Repeat the complete ordered sequence")
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--output-file", required=True)
    parser.add_argument(
        "--server-configuration", default="unspecified", help="Free-form label, e.g. eager or regional-dynamic"
    )
    parser.add_argument("--fail-fast", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1")
    report = run(args)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    summary = report["summary"]
    latency = summary["latency"]
    print(
        f"completed={summary['successful']}/{summary['requests']} "
        f"p50={latency['p50_s']:.3f}s p100={latency['p100_s']:.3f}s output={output_path}"
    )
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
