#!/usr/bin/env python3
"""Measure one SwiftLM server: TTFT, decode tok/s, output hash per prompt.

Streams /v1/chat/completions at temperature 0 with a fixed seed. Decode tok/s
is (completion_tokens - 1) / (t_last_token - t_first_token), so prefill time
is excluded. The output hash is for checking that runs give the same output.
"""
import argparse
import hashlib
import json
import statistics
import sys
import time
import urllib.request

PROMPTS = [
    "Explain, step by step, how a B-tree insertion splits a full node.",
    "Write a Python function that returns the n-th Fibonacci number using matrix exponentiation, then explain its complexity.",
    "Summarize the causes of the 1929 stock market crash in five bullet points.",
]


def scrape_metrics(base):
    out = {}
    try:
        with urllib.request.urlopen(base + "/metrics", timeout=10) as r:
            for line in r.read().decode().splitlines():
                if line.startswith("swiftlm_") and " " in line:
                    k, v = line.rsplit(" ", 1)
                    out[k] = float(v)
    except Exception as e:  # metrics are best effort
        out["error"] = str(e)
    return out


def run_prompt(base, model, prompt, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 1234,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    t_first = t_last = None
    text, chunks, usage = [], 0, None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                if piece:
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    chunks += 1
                    text.append(piece)
    out = "".join(text)
    n = (usage or {}).get("completion_tokens", chunks)
    decode_s = (t_last - t_first) if t_first and t_last else 0.0
    return {
        "ttft_s": (t_first - t0) if t_first else None,
        "completion_tokens": n,
        "decode_tok_s": (n - 1) / decode_s if decode_s > 0 and n > 1 else None,
        "sha256": hashlib.sha256(out.encode()).hexdigest()[:16],
        "chars": len(out),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5413")
    ap.add_argument("--model", default="local")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    for _ in range(args.warmup):
        run_prompt(args.base, args.model, "Say hi.", 8)

    results = []
    for run in range(args.runs):
        for i, p in enumerate(PROMPTS):
            r = run_prompt(args.base, args.model, p, args.max_tokens)
            r.update(run=run, prompt=i)
            results.append(r)
            print(f"  [{args.label}] run={run} prompt={i} ttft={r['ttft_s']:.2f}s "
                  f"decode={r['decode_tok_s'] or 0:.2f} tok/s n={r['completion_tokens']} "
                  f"hash={r['sha256']}", file=sys.stderr)

    rates = [r["decode_tok_s"] for r in results if r["decode_tok_s"]]
    hashes = {}
    for r in results:
        hashes.setdefault(r["prompt"], set()).add(r["sha256"])
    summary = {
        "label": args.label,
        "decode_tok_s_median": statistics.median(rates) if rates else None,
        "decode_tok_s_min": min(rates) if rates else None,
        "decode_tok_s_max": max(rates) if rates else None,
        "ttft_s_median": statistics.median([r["ttft_s"] for r in results if r["ttft_s"]]),
        "deterministic": all(len(h) == 1 for h in hashes.values()),
        "hashes": {k: sorted(v) for k, v in hashes.items()},
        "metrics": scrape_metrics(args.base),
        "runs": results,
    }
    json.dump(summary, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
