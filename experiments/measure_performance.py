"""Manually measure completed SSE events; TPOT uses a character approximation."""

import argparse
import json
from pathlib import Path
import time
from experiments._shared import RESULTS_DIR, save_json


def iter_sse_events(lines):
    """Decode SSE event blocks, comments and multiline data without network IO."""
    event, data = "message", []
    for raw in lines:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = "message", []
        elif line.startswith("event:"):
            event = line[6:].lstrip()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield event, "\n".join(data)


def collect_metrics(events, start, clock=time.perf_counter):
    """Consume (event, data) pairs; timestamps are event arrival times."""
    first_event = first_answer = last_answer = None
    characters = chunks = 0
    error = None
    completed = False
    for event, data in events:
        arrived = clock()
        if first_event is None:
            first_event = arrived
        if event == "done" or data == "[DONE]":
            completed = True
            break
        if event == "error":
            error = data
            continue
        if event in {"answer_chunk", "answer"}:
            text = json.loads(data)
            if isinstance(text, str) and text:
                if first_answer is None:
                    first_answer = arrived
                last_answer = arrived
                characters += len(text)
                chunks += 1
    end = clock()
    tokens = characters / 1.5
    generation = last_answer-first_answer if chunks > 1 else None
    return {
        "completed": completed, "error": error,
        "total_seconds": end-start,
        "system_ttft_seconds": first_event-start if first_event is not None else None,
        "answer_ttfa_seconds": first_answer-start if first_answer is not None else None,
        "generation_seconds": generation,
        "answer_characters": characters, "answer_chunks": chunks,
        "estimated_tokens": tokens,
        "estimated_tpot_ms": generation/tokens*1000 if generation is not None and tokens else None,
        "token_estimation": "characters / 1.5; approximate, not tokenizer or provider token timing",
    }


def measure_performance(url, message, timeout=120.0):
    import requests
    start = time.perf_counter()
    with requests.post(url, json={"message": message}, stream=True, timeout=(10, timeout)) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        return collect_metrics(iter_sse_events(response.iter_lines(decode_unicode=True)), start)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/chat")
    parser.add_argument("--message", default="演示系统如何结合关键词检索和向量检索？")
    parser.add_argument("--timeout", type=float, default=120.0, help="Read timeout in seconds")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "performance.json")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    metrics = measure_performance(args.url, args.message, args.timeout)
    save_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0 if metrics["completed"] and not metrics["error"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
