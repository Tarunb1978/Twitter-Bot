"""
Offline evaluator for premium dispatch quality and style drift.

Usage:
    python quality/evaluate_dispatch.py --text "HEADLINE: ..."
    python quality/evaluate_dispatch.py --file path/to/dispatch.txt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


INSTITUTIONAL_TOKENS = [
    "institutional",
    "de-risking",
    "positioning unwind",
    "risk aversion",
    "systematic hedge funds",
    "liquidity",
    "rotation",
]

BANNED = [
    "[act",
    "act 1",
    "act 2",
    "act 3",
    "summary",
    "the real question is",
    "it's interesting to note",
]


def evaluate(text: str) -> dict:
    text = (text or "").strip()
    lowered = text.lower()
    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    body_parts = parts[1:] if parts and parts[0].startswith("HEADLINE:") else parts
    numeric_tokens = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", text)

    failed = []
    if not text.startswith("HEADLINE:"):
        failed.append("missing_headline")
    if len(body_parts) < 3 or len(body_parts) > 4:
        failed.append("body_paragraph_count_out_of_range")
    if len(numeric_tokens) < 6:
        failed.append("low_metric_density")
    if not any(t in lowered for t in INSTITUTIONAL_TOKENS):
        failed.append("missing_institutional_framing")
    for b in BANNED:
        if b in lowered:
            failed.append(f"banned_phrase:{b}")

    watch_grid_ok = False
    if body_parts:
        tail = body_parts[-1].lower()
        watch_grid_ok = (
            "trigger:" in tail and "confirmation:" in tail and "invalidation:" in tail
        )
        if not watch_grid_ok:
            failed.append("missing_watch_grid")

    return {
        "passed": len(failed) == 0,
        "failed_rules": failed,
        "metrics": {
            "paragraph_count": len(body_parts),
            "numeric_token_count": len(numeric_tokens),
            "watch_grid_ok": watch_grid_ok,
        },
    }


def load_text(args: argparse.Namespace) -> str:
    if args.text:
        return args.text
    if args.file:
        return Path(args.file).read_text(encoding="utf-8")
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="")
    parser.add_argument("--file", type=str, default="")
    opts = parser.parse_args()

    text = load_text(opts)
    result = evaluate(text)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
