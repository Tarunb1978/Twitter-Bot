"""
Indian Markets Tweet Agent -- Entry Point

Runs the LangGraph agent that:
1. Fetches post-market data (indices, top movers, global context)
2. Fetches financial news from MarketAux + NewsData.io
3. Generates 5 Kobeissi Letter-style tweet drafts
4. Audits each tweet for accuracy, tone, and relevance (scores 1-10)
5. Emails the scored drafts to your inbox for review

Usage:
    python main.py
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime

from agent import build_graph
from config import EMAIL_DELIVERY, PRINT_FINAL_TO_TERMINAL, RUN_LOCK_STALE_MINUTES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            f"logs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        ),
    ],
)
logger = logging.getLogger(__name__)
LOCK_PATH = Path("logs/run.lock")


def _read_lock_pid_and_started() -> tuple[int | None, datetime | None]:
    try:
        text = LOCK_PATH.read_text(encoding="utf-8")
        pid_val: int | None = None
        started: datetime | None = None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("pid="):
                rest = line[4:].split()[0]
                pid_val = int(rest)
            elif line.startswith("started_at="):
                started = datetime.fromisoformat(line.split("=", 1)[1].strip())
        return pid_val, started
    except Exception:
        return None, None


def _process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _break_stale_run_lock() -> None:
    """
    Remove run.lock if the owning process is gone or the lock is older than RUN_LOCK_STALE_MINUTES.
    Prevents permanent deadlock after Ctrl+C, IDE abort, or crash (finally did not run).
    """
    if not LOCK_PATH.exists():
        return
    pid, started = _read_lock_pid_and_started()
    should_break = False
    reason = ""
    if pid is not None and not _process_exists(pid):
        should_break = True
        reason = f"pid {pid} not running"
    elif started is not None:
        age_min = (datetime.now() - started).total_seconds() / 60.0
        if age_min > RUN_LOCK_STALE_MINUTES:
            should_break = True
            reason = f"lock age {age_min:.1f}m > {RUN_LOCK_STALE_MINUTES}m"
    if should_break:
        try:
            LOCK_PATH.unlink(missing_ok=True)
            logger.warning("Removed stale run.lock (%s).", reason)
        except Exception:
            pass


def _acquire_run_lock() -> bool:
    """
    Acquire a single-run lock to prevent overlapping executions.
    Uses O_EXCL create semantics, which is atomic on Windows.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    _break_stale_run_lock()
    try:
        fd = os.open(str(LOCK_PATH), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(f"pid={os.getpid()} started_at={datetime.now().isoformat()}\n")
        return True
    except FileExistsError:
        return False


def _release_run_lock() -> None:
    try:
        LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _configure_stdout_utf8() -> None:
    reconf = getattr(sys.stdout, "reconfigure", None)
    if callable(reconf):
        try:
            reconf(encoding="utf-8", errors="replace")
        except Exception:
            pass


def _emit_final_tweets_terminal_only(final: list) -> None:
    """
    Full tweet text goes to stdout only (VS Code / integrated terminal).
    Log files get a one-line summary with no draft body.
    """
    should_print = EMAIL_DELIVERY == "terminal" or PRINT_FINAL_TO_TERMINAL
    n = len(final) if final else 0
    if should_print:
        logger.info("Final output (%d part(s)) - full text below on terminal only.", n)
    else:
        logger.info(
            "Final output (%d part(s)); body not printed (set EMAIL_DELIVERY=terminal or PRINT_FINAL_TO_TERMINAL=true).",
            n,
        )

    if not should_print or not final:
        return

    _configure_stdout_utf8()
    for t in final:
        tt = t.get("type", "") or "unknown"
        raw_body = t.get("tweet")
        body = "" if raw_body is None else str(raw_body)
        if tt == "premium_thread":
            print("\n=== PREMIUM THREAD (full) ===\n", body, sep="", flush=True)
        elif tt == "standalone":
            print("\n=== STANDALONE TWEET (full) ===\n", body, sep="", flush=True)
        else:
            print("\n=== FINAL OUTPUT (", tt, ") ===\n", body, sep="", flush=True)


def main():
    if not _acquire_run_lock():
        logger.error(
            "Another run is already in progress (run.lock held by an active process). "
            "If that is wrong, delete logs/run.lock or wait for it to finish."
        )
        sys.exit(2)

    logger.info("Starting Tweet Agent...")

    graph = build_graph()

    initial_state = {
        "market_data": {},
        "macro_data": {},
        "news_articles": [],
        "draft_tweets": [],
        "scored_tweets": [],
        "approved_tweets": [],
        "final_tweets": [],
        "story_candidates": [],
        "selected_stories": [],
        "coverage_telemetry": {},
        "prioritizer_memory": {},
        "dispatch_qa": {},
        "audit_feedback": "",
        "attempt": 0,
    }

    try:
        result = graph.invoke(initial_state)

        attempt = result.get("attempt", 1)
        final = result.get("final_tweets", [])
        approved_count = len(result.get("approved_tweets", []))

        logger.info(f"Completed in {attempt} batch(es), {approved_count} tweets approved")

        _emit_final_tweets_terminal_only(final)

        dispatch_qa = result.get("dispatch_qa", {})
        if dispatch_qa:
            logger.info(
                "Dispatch QA | dispatch_validation_passed=%s | failed_rules=%s | hard_fail_reasons=%s | qa_v2_scores=%s | weighted_total=%s | rewrite_attempted=%s | rewrite_passed=%s | rewrite_reason_codes=%s | regime=%s",
                dispatch_qa.get("dispatch_validation_passed"),
                dispatch_qa.get("failed_rules", []),
                dispatch_qa.get("hard_fail_reasons", []),
                dispatch_qa.get("qa_v2_scores", {}),
                dispatch_qa.get("weighted_total"),
                dispatch_qa.get("rewrite_attempted", False),
                dispatch_qa.get("rewrite_passed", False),
                dispatch_qa.get("rewrite_reason_codes", []),
                result.get("regime", "unknown"),
            )

        coverage_telemetry = result.get("coverage_telemetry", {})
        if coverage_telemetry:
            logger.info(
                "Coverage | mode=%s | selected_themes=%s | selected_count=%s | rejected=%s | novelty_penalties=%s | tie_break_applied=%s",
                coverage_telemetry.get("prioritizer_mode"),
                coverage_telemetry.get("selected_theme_mix", {}),
                coverage_telemetry.get("selected_count", 0),
                coverage_telemetry.get("rejected_summary", {}),
                coverage_telemetry.get("novelty_penalties", []),
                coverage_telemetry.get("tie_break_applied", False),
            )

        logger.info(f"Email status: {result.get('audit_feedback', 'unknown')}")

    except Exception:
        logger.exception("Agent failed")
        sys.exit(1)
    finally:
        _release_run_lock()


if __name__ == "__main__":
    main()
