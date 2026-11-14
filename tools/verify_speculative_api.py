#!/usr/bin/env python3
"""Verify actual single-stream HTTP/SSE throughput and original 8B token IDs."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from verify_generation_target import CASES

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, default=ROOT / "result/bin/plamo-server")
    parser.add_argument("--source", action="store_true", help="use tools/server.py --cache")
    parser.add_argument("--worker", type=Path)
    parser.add_argument(
        "--reference-report", type=Path, default=ROOT / "docs/benchmarks/throughput-target.json"
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--target", type=float, default=10.0)
    parser.add_argument("--first-direction", choices=("en-ja", "ja-en"), default="en-ja")
    parser.add_argument(
        "--ttft-target",
        type=float,
        help="maximum HTTP-to-first-visible-text time in seconds, including the first request after ready",
    )
    parser.add_argument("--report", type=Path, default=ROOT / "build/speculative-api.json")
    args = parser.parse_args()
    if (
        args.repeats < 1
        or args.target <= 0
        or (args.ttft_target is not None and args.ttft_target <= 0)
    ):
        parser.error("repeats and targets must be positive")
    reference = json.loads(args.reference_report.read_text())
    if "final_worker_verification" in reference:
        reference = reference["final_worker_verification"]
    expected = {s["direction"]: s for s in reference["runs"]["reference"]["samples"]}
    cases = CASES if args.first_direction == "en-ja" else tuple(reversed(CASES))
    worker_digest = hashlib.sha256(args.worker.read_bytes()).hexdigest() if args.worker else None
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    headers = {"Content-Type": "application/json"}
    if os.environ.get("PLAMO_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["PLAMO_API_KEY"]

    def health():
        req = urllib.request.Request(base + "/health", headers=headers)
        try:
            response = urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as failure:
            with failure:
                return json.load(failure)
        with response:
            return json.load(response)

    def translate(direction, text):
        body = dict(
            text=text,
            source_lang="en" if direction == 1 else "ja",
            target_lang="ja" if direction == 1 else "en",
            max_tokens=64,
            stream=True,
        )
        req = urllib.request.Request(
            base + "/translate", data=json.dumps(body).encode(), headers=headers
        )
        ids, pieces, arrivals, kinds = [], [], [], []
        result, kind, first_text = None, None, None
        start = time.monotonic()
        with urllib.request.urlopen(req, timeout=300) as response:
            for raw in response:
                line = raw.decode().rstrip("\r\n")
                if line.startswith("event: "):
                    kind = line[7:]
                elif line.startswith("data: "):
                    event = json.loads(line[6:])
                    kinds.append(kind)
                    if kind == "token":
                        pieces.append(event["text"])
                        if event["text"] and first_text is None:
                            first_text = time.monotonic() - start
                        if "token_id" in event:
                            ids.append(event["token_id"])
                            arrivals.append(time.monotonic() - start)
                    elif kind == "done":
                        result = event
        assert result and kinds[0] == "prefill" and kinds[-1] == "done", kinds
        assert first_text is not None, "No visible translation text was received"
        assert (
            "".join(pieces) == result["translation"]
            and len(ids) == result["usage"]["completion_tokens"]
        )
        timing, speculative = result["timings"], result["speculative"]
        assert (
            ids == expected[direction]["token_ids"]
            and result["translation"] == expected[direction]["translation"]
        )
        assert timing["decode_steps"] == speculative["consumed_tokens"] == len(ids) - 1
        assert (
            speculative["batches"] > 0
            and timing["decode_command_buffers"] >= speculative["batches"]
        )
        assert result["compute"]["precision"] == "fp32" and result["compute"]["weight_storage"] in (
            "bf16",
            "lossless-bf16",
            "texture-bf16",
        )
        assert not result["compute"]["errors"] and not result["compute"]["cpu_fallbacks"]
        return dict(
            direction=direction,
            input=text,
            wall_seconds=time.monotonic() - start,
            first_text_seconds=first_text,
            tokens_per_second=timing["decode_steps"] / timing["decode_seconds"],
            wall_inter_token_tokens_per_second=(len(arrivals) - 1) / (arrivals[-1] - arrivals[0]),
            token_ids=ids,
            arrivals=arrivals,
            result=result,
        )

    command = (
        [sys.executable, str(ROOT / "tools/server.py"), "--cache"]
        if args.source
        else [str(args.server.resolve())]
    )
    command += ["--model", "8b", "--port", str(port)]
    if args.worker:
        command += ["--worker", str(args.worker.resolve())]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 180
            while True:
                assert process.poll() is None, "API process exited; see log"
                try:
                    ready = health()
                    if ready["status"] == "ready":
                        break
                    assert ready["status"] == "loading", ready
                except OSError:
                    pass
                assert time.monotonic() < deadline, "API readiness timed out"
                time.sleep(0.2)
            assert ready["compute"]["backend"] == "metal" and ready["speculative"]["enabled"], ready
            samples = []
            # No warmup or cached answers: measure the first actual request
            # after readiness, then serialized English/Japanese requests.
            for trial in range(args.repeats):
                for direction, text in cases:
                    sample = dict(trial=trial + 1, **translate(direction, text))
                    samples.append(sample)
                    args.report.with_suffix(".partial.json").write_text(
                        json.dumps(samples, ensure_ascii=False, indent=2) + "\n"
                    )
                    print(
                        f"{trial + 1} {direction}: {sample['tokens_per_second']:.3f} tokens/s; "
                        f"SSE wall {sample['wall_inter_token_tokens_per_second']:.3f}; "
                        f"first text {sample['first_text_seconds']:.3f}s",
                        flush=True,
                    )
            final = health()
            assert (
                final["status"] == "ready"
                and not final["busy"]
                and final["worker_pid"] == ready["worker_pid"]
            )
            passed = all(
                s["tokens_per_second"] > args.target
                and s["wall_inter_token_tokens_per_second"] > args.target
                for s in samples
            )
            ttft_passed = (
                all(s["first_text_seconds"] <= args.ttft_target for s in samples)
                if args.ttft_target is not None
                else None
            )
            assert (
                not args.worker
                or hashlib.sha256(args.worker.read_bytes()).hexdigest() == worker_digest
            ), "Worker changed during measurement"
            report = dict(
                timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                package=None if args.source else str(args.server.resolve()),
                source_server=args.source,
                worker_override=str(args.worker) if args.worker else None,
                worker_sha256=worker_digest,
                first_direction=args.first_direction,
                ready=ready,
                warmup=None,
                target_tokens_per_second=args.target,
                samples=samples,
                final_health=final,
                all_samples_exceed_target=passed,
                exact_reference_tokens_and_text=True,
                ttft_target_seconds=args.ttft_target,
                all_samples_meet_ttft_target=ttft_passed,
                environment={
                    k: os.environ.get(k, "auto")
                    for k in ("PLAMO_WEIGHT_STORAGE", "PLAMO_METAL_CACHE", "PLAMO_SPECULATIVE")
                },
                limits="One persistent API worker, serialized single streams, 64 output tokens per request. "
                "Decode excludes startup/prefill. Wall rate uses SSE token arrivals. OS caching and other apps are uncontrolled.",
            )
            args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            assert passed, "Actual API did not exceed target; see report"
            assert ttft_passed is not False, "Actual API exceeded the TTFT target; see report"
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
