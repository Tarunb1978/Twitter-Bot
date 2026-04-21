"""
LangGraph agent -- the brain of the tweet bot.

Batch accumulation graph:
  Writer generates 3 tweets/batch -> Auditor scores them ->
  approved tweets (>= 10) accumulate -> loop until 5 good tweets collected.
"""

import json
import hashlib
import logging
import os
import random
import re
import threading
import time
from collections import Counter
from datetime import datetime
from typing import Any, Literal, TypedDict

try:
    from openai import RateLimitError
except ImportError:

    class RateLimitError(Exception):
        """Unused placeholder if openai is not installed."""

        pass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import END, StateGraph

from config import (
    AUDIT_PASS_THRESHOLD,
    AUDITOR_MODEL,
    EMAIL_DELIVERY,
    LLM_PROVIDER,
    MAX_AUDIT_CYCLES,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_NUM_CTX,
    OPENAI_API_KEY,
    OPENAI_MIN_CALL_INTERVAL_SEC,
    OPENAI_RATE_LIMIT_MAX_ATTEMPTS,
    OPENAI_RETRY_BASE_DELAY_SEC,
    OPENAI_RETRY_MAX_DELAY_SEC,
    OPENAI_SDK_MAX_RETRIES,
    PRIORITIZER_BRENT_ANCHOR_PCT,
    PRIORITIZER_CRUDE_ANCHOR_PCT,
    PRIORITIZER_FX_WEEK_ANCHOR,
    PRIORITIZER_GOLD_ANCHOR_PCT,
    PRIORITIZER_MAX_PER_THEME,
    PRIORITIZER_MEMORY_WINDOW,
    PRIORITIZER_MIN_SCORE,
    PRIORITIZER_MODE,
    PRIORITIZER_THEME_TARGETS,
    PRIORITIZER_TOP_K,
    PRIORITIZER_VIX_ANCHOR_CHANGE_PCT,
    SYNTHETIC_STORY_SCORE_BOOST,
    TOTAL_NEEDED,
    TWEET_BATCH_SIZE,
    WRITER_MODEL,
)
from prompts import (
    AUDITOR_PROMPT,
    OLLAMA_SYNTHESIZER_JSON_SUFFIX,
    PRIORITIZER_TIEBREAKER_PROMPT,
    SYNTHESIZER_PROMPT,
    WRITER_BATCH_TEMPLATE,
    WRITER_PROMPT,
    WRITER_TWEET_CHANNEL_GUIDE,
)
from tools.email import send_email
from tools.market import get_market_data
from tools.macro import get_macro_data
from tools.news import fetch_news

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    market_data: dict
    macro_data: dict
    news_articles: list[dict]
    draft_tweets: list[str]          # Current batch (3 tweets)
    scored_tweets: list[dict]        # Current batch scored
    approved_tweets: list[dict]      # Accumulator -- good tweets across ALL batches
    final_tweets: list[dict]         # Synthesizer output (premium thread + standalone)
    story_candidates: list[dict]     # Prioritizer full candidate list
    selected_stories: list[dict]     # Prioritizer selected shortlist for writer
    coverage_telemetry: dict         # Selection diagnostics and diversity telemetry
    prioritizer_memory: dict         # Rolling run-memory summary
    claim_evidence: dict             # Deterministic claim-evidence graph for synthesis
    regime: str                      # Deterministic market regime classification
    dispatch_qa: dict                # Synthesizer quality gates + retry telemetry
    audit_feedback: str
    attempt: int                     # Batch number (1-based)

# ---------------------------------------------------------------------------
# LLMs -- All chat steps: OpenAI or local Ollama (config LLM_PROVIDER).
# ---------------------------------------------------------------------------

_openai_throttle_lock = threading.Lock()
_openai_last_call_monotonic: float | None = None


def _is_openai_rate_limit_error(exc: BaseException) -> bool:
    seen: set[int] = set()

    def walk(e: BaseException) -> bool:
        if id(e) in seen:
            return False
        seen.add(id(e))
        if isinstance(e, RateLimitError):
            return True
        msg = str(e).lower()
        if "429" in msg and ("rate" in msg or "limit" in msg or "token" in msg or "requests" in msg):
            return True
        if "too many requests" in msg:
            return True
        cause = getattr(e, "__cause__", None)
        if isinstance(cause, BaseException) and cause is not e and walk(cause):
            return True
        return False

    return walk(exc)


def _openai_throttle_before_call() -> None:
    """Minimum spacing between OpenAI HTTP calls (reduces burst 429s)."""
    global _openai_last_call_monotonic
    if LLM_PROVIDER != "openai" or OPENAI_MIN_CALL_INTERVAL_SEC <= 0:
        return
    with _openai_throttle_lock:
        now = time.monotonic()
        if _openai_last_call_monotonic is not None:
            wait = OPENAI_MIN_CALL_INTERVAL_SEC - (now - _openai_last_call_monotonic)
            if wait > 0:
                time.sleep(wait)
        _openai_last_call_monotonic = time.monotonic()


class _ThrottledRetryChatOpenAI(ChatOpenAI):
    """
    Per-call throttle + exponential backoff on 429 / RateLimitError.
    Covers agent tool loops (each model completion goes through invoke/ainvoke).
    """

    def invoke(self, input: Any, config: Any | None = None, *, stop: list[str] | None = None, **kwargs: Any):
        if LLM_PROVIDER != "openai":
            return super().invoke(input, config=config, stop=stop, **kwargs)
        attempts = max(1, OPENAI_RATE_LIMIT_MAX_ATTEMPTS)
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            _openai_throttle_before_call()
            try:
                return super().invoke(input, config=config, stop=stop, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt >= attempts - 1 or not _is_openai_rate_limit_error(e):
                    raise
                delay = min(
                    OPENAI_RETRY_MAX_DELAY_SEC,
                    OPENAI_RETRY_BASE_DELAY_SEC * (2**attempt),
                )
                jitter = random.uniform(0, min(1.5, delay * 0.35))
                logger.warning(
                    "OpenAI rate limited (attempt %s/%s); sleeping %.1fs then retrying.",
                    attempt + 1,
                    attempts,
                    delay + jitter,
                )
                time.sleep(delay + jitter)
        assert last_exc is not None
        raise last_exc

    async def ainvoke(self, input: Any, config: Any | None = None, *, stop: list[str] | None = None, **kwargs: Any):
        if LLM_PROVIDER != "openai":
            return await super().ainvoke(input, config=config, stop=stop, **kwargs)
        import asyncio

        attempts = max(1, OPENAI_RATE_LIMIT_MAX_ATTEMPTS)
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            _openai_throttle_before_call()
            try:
                return await super().ainvoke(input, config=config, stop=stop, **kwargs)
            except Exception as e:
                last_exc = e
                if attempt >= attempts - 1 or not _is_openai_rate_limit_error(e):
                    raise
                delay = min(
                    OPENAI_RETRY_MAX_DELAY_SEC,
                    OPENAI_RETRY_BASE_DELAY_SEC * (2**attempt),
                )
                jitter = random.uniform(0, min(1.5, delay * 0.35))
                logger.warning(
                    "OpenAI rate limited async (attempt %s/%s); sleeping %.1fs then retrying.",
                    attempt + 1,
                    attempts,
                    delay + jitter,
                )
                await asyncio.sleep(delay + jitter)
        assert last_exc is not None
        raise last_exc


def _make_chat_llm(*, model_family: Literal["writer", "auditor"], temperature: float):
    if LLM_PROVIDER == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                "LLM_PROVIDER=ollama requires langchain-ollama. Install: pip install langchain-ollama"
            ) from e
        kwargs: dict = {
            "model": OLLAMA_MODEL,
            "base_url": OLLAMA_BASE_URL,
            "temperature": temperature,
        }
        if OLLAMA_NUM_CTX is not None:
            kwargs["num_ctx"] = OLLAMA_NUM_CTX
        return ChatOllama(**kwargs)
    return _ThrottledRetryChatOpenAI(
        api_key=OPENAI_API_KEY,
        model=WRITER_MODEL if model_family == "writer" else AUDITOR_MODEL,
        temperature=temperature,
        max_retries=OPENAI_SDK_MAX_RETRIES,
    )


writer_llm = _make_chat_llm(model_family="writer", temperature=0.7)
auditor_llm = _make_chat_llm(model_family="auditor", temperature=0.1)
synthesizer_llm = _make_chat_llm(model_family="writer", temperature=0.4)
prioritizer_llm = _make_chat_llm(model_family="auditor", temperature=0.1)

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "logs", "story_memory.json")


def _normalize_headline(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _theme_keywords() -> dict:
    # Order matters: more specific themes before broad India tape.
    return {
        "commodityShock": ["crude", "brent", "oil", "gold", "commodity", "metals", "supply shock", "lng", "opec"],
        "geopoliticsEnergy": ["war", "strait", "hormuz", "west asia", "middle east", "sanction", "fuel crisis", "energy crisis"],
        "policyRepricing": ["rbi", "fed", "policy rate", "repo", "yield", "inflation", "cpi", "mpc"],
        "fxStress": ["usd/inr", "rupee", "inr", "dollar index", "dxy", "fx", "carry trade", "currency"],
        "riskOffUnwind": ["vix", "risk-off", "volatility", "selloff", "de-risk", "liquidation", "margin call", "flash crash"],
        "indiaEquityTape": ["nifty", "sensex", "nifty 50", "bank nifty", "midcap", "smallcap", "fii", "fpi", "dii", "bse", "nse"],
        "idiosyncraticCorporate": ["adani", "results", "guidance", "deal", "order book", "52-week", "52 week", "default", "probe", "sebi", "merger", "acquisition"],
    }


def _infer_story_theme(article: dict, market_data: dict, macro_data: dict) -> str:
    text = f"{article.get('title','')} {article.get('summary','')}".lower()
    for theme, words in _theme_keywords().items():
        if any(w in text for w in words):
            return theme
    global_data = market_data.get("global", {})
    vix = market_data.get("india_vix") or {}
    brent = abs(float((global_data.get("Brent") or {}).get("change_pct", 0) or 0))
    gold = abs(float((global_data.get("Gold") or {}).get("change_pct", 0) or 0))
    crude = abs(float((global_data.get("Crude Oil") or {}).get("change_pct", 0) or 0))
    fx_week = abs(float((macro_data.get("usdinr") or {}).get("week_change", 0) or 0))
    vix_ch = abs(float(vix.get("change_pct", 0) or 0))
    if brent >= PRIORITIZER_BRENT_ANCHOR_PCT or gold >= PRIORITIZER_GOLD_ANCHOR_PCT or crude >= PRIORITIZER_CRUDE_ANCHOR_PCT:
        return "commodityShock"
    if fx_week >= PRIORITIZER_FX_WEEK_ANCHOR:
        return "fxStress"
    if vix_ch >= PRIORITIZER_VIX_ANCHOR_CHANGE_PCT:
        return "riskOffUnwind"
    movers = market_data.get("top_movers", {})
    breadth = movers.get("breadth", {})
    green = int(breadth.get("green", 0) or 0)
    total = int(breadth.get("total", 0) or 0)
    if total > 0 and green / total < 0.25:
        return "riskOffUnwind"
    return "indiaEquityTape"


def _extract_hard_facts_for_theme(
    theme: str,
    article: dict | None,
    market_data: dict,
    macro_data: dict,
    primary_global: str | None = None,
) -> list[str]:
    facts: list[str] = []
    if article and article.get("title"):
        facts.append(f"Headline: {str(article['title'])[:200]}")
    g = market_data.get("global") or {}
    indices = market_data.get("indices") or {}
    sectors = market_data.get("sectors") or {}
    vix = market_data.get("india_vix") or {}
    movers = market_data.get("top_movers") or {}

    def add_idx(name: str) -> None:
        row = indices.get(name) or {}
        if row.get("price") is not None and row.get("change_pct") is not None:
            facts.append(f"{name} {row['change_pct']:+.2f}% to {row['price']}")

    def add_global(name: str) -> None:
        row = g.get(name) or {}
        if row.get("price") is not None and row.get("change_pct") is not None:
            facts.append(f"{name} {row['change_pct']:+.2f}% at {row['price']}")

    def add_sector(name: str) -> None:
        row = sectors.get(name) or {}
        if row.get("price") is not None and row.get("change_pct") is not None:
            facts.append(f"{name} {row['change_pct']:+.2f}% to {row['price']}")

    if primary_global == "Brent":
        add_global("Brent")
        add_global("Crude Oil")
        add_global("Gold")
        add_idx("Nifty 50")
    elif primary_global == "Gold":
        add_global("Gold")
        add_global("Brent")
        add_global("Crude Oil")
        add_idx("Sensex")
    elif theme == "commodityShock":
        add_global("Brent")
        add_global("Crude Oil")
        add_global("Gold")
        add_idx("Nifty 50")
        add_idx("Sensex")
    elif theme == "fxStress":
        usd = macro_data.get("usdinr") or {}
        if usd.get("spot_rate") is not None:
            facts.append(f"USD/INR spot {usd['spot_rate']}")
        if usd.get("week_change") is not None:
            facts.append(f"USD/INR week_change {usd['week_change']}")
        add_global("USD/INR")
        add_idx("Nifty 50")
        add_global("Gold")
    elif theme in ("riskOffUnwind", "indiaEquityTape"):
        if vix.get("value") is not None and vix.get("change_pct") is not None:
            facts.append(f"India VIX {vix['change_pct']:+.2f}% to {vix['value']}")
        add_idx("Nifty 50")
        add_idx("Sensex")
        breadth = movers.get("breadth", {})
        if breadth.get("green") is not None and breadth.get("total") is not None:
            facts.append(f"breadth green {breadth['green']} total {breadth['total']}")
        add_global("Brent")
        add_global("Gold")
    else:
        add_idx("Nifty 50")
        add_idx("Sensex")
        for n in ("Nifty Bank", "Nifty IT", "Nifty Pharma", "Nifty FMCG"):
            add_sector(n)
        for s in (movers.get("gainers") or [])[:2]:
            if isinstance(s, dict) and s.get("change_pct") is not None:
                facts.append(f"gainer {s.get('ticker','')} {s['change_pct']:+.2f}%")
        for s in (movers.get("losers") or [])[:2]:
            if isinstance(s, dict) and s.get("change_pct") is not None:
                facts.append(f"loser {s.get('ticker','')} {s['change_pct']:+.2f}%")
        add_global("Brent")
        add_global("S&P 500")

    usdinr = macro_data.get("usdinr") or {}
    if theme not in ("fxStress",) and usdinr.get("spot_rate") is not None and len(facts) < 10:
        facts.append(f"USD/INR spot {usdinr['spot_rate']}")

    return facts[:12]


def _build_transmission_path(theme: str, market_data: dict, macro_data: dict) -> str:
    mapping = {
        "commodityShock": "Global commodity move -> India import/cost channel -> sector margin repricing",
        "fxStress": "FX volatility -> INR sensitivity -> exporter/importer divergence",
        "policyRepricing": "Rates/policy repricing -> liquidity/carry economics -> sector rotation",
        "riskOffUnwind": "Volatility spike -> de-risking/liquidation -> breadth deterioration",
        "indiaEquityTape": "India cash index move -> flow/breadth -> sector beta and positioning",
        "idiosyncraticCorporate": "Company catalyst -> positioning/flow response -> index/sector spillover",
        "geopoliticsEnergy": "Geopolitical shock -> energy risk premium -> India inflation/current account stress",
    }
    return mapping.get(theme, "Macro shock -> India transmission -> portfolio reallocation")


def _score_story_candidate(candidate: dict, market_data: dict, macro_data: dict, memory: dict | None = None) -> dict:
    breakdown = {
        "impact": 0,
        "india_transmission": 0,
        "cross_asset_confirmation": 0,
        "timeliness": 0,
        "credibility": 0,
        "novelty_proxy": 0,
        "allocation_penalty": 0,
    }
    theme = candidate.get("theme", "")
    text = f"{candidate.get('headline','')} {candidate.get('summary','')}".lower()
    global_data = market_data.get("global", {})
    vix = market_data.get("india_vix") or {}
    movers = market_data.get("top_movers", {})
    breadth = movers.get("breadth", {})
    brent_move = abs(float((global_data.get("Brent") or {}).get("change_pct", 0) or 0))
    gold_move = abs(float((global_data.get("Gold") or {}).get("change_pct", 0) or 0))
    fx_week = abs(float((macro_data.get("usdinr") or {}).get("week_change", 0) or 0))
    vix_move = abs(float(vix.get("change_pct", 0) or 0))
    green = int(breadth.get("green", 0) or 0)
    total = int(breadth.get("total", 0) or 0)
    weak_breadth = total > 0 and green / total < 0.3

    if brent_move >= 3 or gold_move >= 2 or vix_move >= 10:
        breakdown["impact"] = 4
    elif brent_move >= 2 or gold_move >= 1.2 or vix_move >= 6:
        breakdown["impact"] = 3
    elif "52-week" in text or "52 week" in text:
        breakdown["impact"] = 2
    else:
        breakdown["impact"] = 1

    if theme in {"commodityShock", "fxStress", "policyRepricing", "geopoliticsEnergy"}:
        breakdown["india_transmission"] = 3
    elif theme in {"riskOffUnwind", "idiosyncraticCorporate", "indiaEquityTape"}:
        breakdown["india_transmission"] = 2

    confirmations = 0
    confirmations += 1 if brent_move >= 2 else 0
    confirmations += 1 if fx_week >= 0.6 else 0
    confirmations += 1 if weak_breadth or vix_move >= 8 else 0
    breakdown["cross_asset_confirmation"] = min(3, confirmations)

    breakdown["timeliness"] = 2 if candidate.get("published") else 1
    breakdown["credibility"] = min(2, max(1, len(candidate.get("sources", []))))
    breakdown["novelty_proxy"] = 1 if len(candidate.get("headline", "")) >= 30 else 0

    if memory:
        recent = memory.get("recent_theme_counts", {})
        target = PRIORITIZER_THEME_TARGETS.get(theme)
        if target is not None:
            total_recent = sum(recent.values()) or 1
            observed = recent.get(theme, 0) / total_recent
            if observed > target + 0.10:
                breakdown["allocation_penalty"] = 1

    total_score = (
        breakdown["impact"]
        + breakdown["india_transmission"]
        + breakdown["cross_asset_confirmation"]
        + breakdown["timeliness"]
        + breakdown["credibility"]
        + breakdown["novelty_proxy"]
        - breakdown["allocation_penalty"]
    )
    return {"total": total_score, "breakdown": breakdown}


def _dedupe_story_candidates(candidates: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for c in candidates:
        key = _normalize_headline(c.get("headline", ""))
        if key not in seen:
            seen[key] = c
        else:
            seen[key]["sources"].extend(c.get("sources", []))
    return list(seen.values())


def _load_prioritizer_memory() -> dict:
    try:
        if os.path.exists(MEMORY_PATH):
            with open(MEMORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {"runs": []}


def _save_prioritizer_memory(memory: dict) -> None:
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=False, indent=2)


def _memory_summary(memory: dict) -> dict:
    runs = (memory or {}).get("runs", [])[-PRIORITIZER_MEMORY_WINDOW:]
    theme_counter = Counter()
    fingerprints = set()
    for run in runs:
        theme_counter.update(run.get("themes", []))
        fingerprints.update(run.get("fingerprints", []))
    return {"recent_theme_counts": dict(theme_counter), "recent_fingerprints": list(fingerprints)}


def _select_diverse_stories(ranked: list[dict]) -> tuple[list[dict], dict]:
    """
    Pick stories across channels: India tape/vol, commodities, then FX/policy/geo/corporate.
    """
    eligible = [c for c in ranked if c.get("score_total", 0) >= PRIORITIZER_MIN_SCORE]
    if len(eligible) < 3:
        eligible = sorted(ranked, key=lambda x: x.get("score_total", 0), reverse=True)

    selected: list[dict] = []
    seen: set[str] = set()
    rejected_summary = {"below_min": 0, "theme_cap": 0, "slots_filled": []}

    def theme_count(th: str) -> int:
        return sum(1 for s in selected if s.get("theme") == th)

    def try_take(predicate, slot_name: str) -> bool:
        for c in sorted(eligible, key=lambda x: x.get("score_total", 0), reverse=True):
            sid = c.get("story_id", "")
            if sid in seen or not sid:
                continue
            if not predicate(c):
                continue
            th = c.get("theme", "unknown")
            if theme_count(th) >= PRIORITIZER_MAX_PER_THEME:
                rejected_summary["theme_cap"] += 1
                continue
            selected.append(c)
            seen.add(sid)
            rejected_summary["slots_filled"].append(slot_name)
            return True
        return False

    try_take(lambda c: c.get("theme") in ("indiaEquityTape", "riskOffUnwind"), "india_vol")
    try_take(lambda c: c.get("theme") == "commodityShock", "commodity")
    try_take(
        lambda c: c.get("theme") in ("fxStress", "policyRepricing", "geopoliticsEnergy", "idiosyncraticCorporate"),
        "fx_policy_geo_corp",
    )

    for c in sorted(eligible, key=lambda x: x.get("score_total", 0), reverse=True):
        if len(selected) >= PRIORITIZER_TOP_K:
            break
        sid = c.get("story_id", "")
        if sid in seen:
            continue
        th = c.get("theme", "unknown")
        if c.get("score_total", 0) < PRIORITIZER_MIN_SCORE:
            rejected_summary["below_min"] += 1
            continue
        if theme_count(th) >= PRIORITIZER_MAX_PER_THEME:
            rejected_summary["theme_cap"] += 1
            continue
        selected.append(c)
        seen.add(sid)

    mix = dict(Counter([s.get("theme", "unknown") for s in selected]))
    return selected, {"selected_theme_mix": mix, "rejected_summary": rejected_summary}


def _build_synthetic_macro_candidates(market_data: dict, macro_data: dict) -> list[dict]:
    """Tool-anchored rows when commodities, VIX, or FX move sharply (ensures non-headline coverage)."""
    out: list[dict] = []
    global_data = market_data.get("global") or {}
    vix = market_data.get("india_vix") or {}

    def push(story_id: str, headline: str, theme: str, primary: str | None) -> None:
        cand = {
            "story_id": story_id,
            "headline": headline,
            "summary": "",
            "theme": theme,
            "published": "",
            "sources": [],
            "hard_facts": _extract_hard_facts_for_theme(theme, None, market_data, macro_data, primary_global=primary),
            "transmission_path": _build_transmission_path(theme, market_data, macro_data),
            "confidence": "high",
            "fingerprint": hashlib.md5(story_id.encode("utf-8")).hexdigest()[:12],
            "synthetic_anchor": True,
        }
        sc = _score_story_candidate(cand, market_data, macro_data, None)
        cand["score_total"] = sc["total"] + SYNTHETIC_STORY_SCORE_BOOST
        cand["score_breakdown"] = sc["breakdown"]
        out.append(cand)

    brent = global_data.get("Brent") or {}
    br = abs(float(brent.get("change_pct", 0) or 0))
    if br >= PRIORITIZER_BRENT_ANCHOR_PCT and brent.get("change_pct") is not None:
        push(
            "synthetic_brent_anchor",
            f"DATA-ANCHOR: Brent {brent['change_pct']:+.2f}% (India import / margin channel)",
            "commodityShock",
            "Brent",
        )

    gold = global_data.get("Gold") or {}
    gg = abs(float(gold.get("change_pct", 0) or 0))
    if gg >= PRIORITIZER_GOLD_ANCHOR_PCT and gold.get("change_pct") is not None:
        push(
            "synthetic_gold_anchor",
            f"DATA-ANCHOR: Gold {gold['change_pct']:+.2f}% (risk-premium / FX read-through)",
            "commodityShock",
            "Gold",
        )

    crude = global_data.get("Crude Oil") or {}
    cr = abs(float(crude.get("change_pct", 0) or 0))
    if cr >= PRIORITIZER_CRUDE_ANCHOR_PCT and crude.get("change_pct") is not None and br < PRIORITIZER_BRENT_ANCHOR_PCT:
        push(
            "synthetic_crude_anchor",
            f"DATA-ANCHOR: Crude Oil {crude['change_pct']:+.2f}%",
            "commodityShock",
            None,
        )

    vix_ch = float(vix.get("change_pct", 0) or 0)
    if abs(vix_ch) >= PRIORITIZER_VIX_ANCHOR_CHANGE_PCT and vix.get("value") is not None:
        push(
            "synthetic_vix_anchor",
            f"DATA-ANCHOR: India VIX {vix_ch:+.2f}% to {vix.get('value')} (tail-risk pricing)",
            "riskOffUnwind",
            None,
        )

    usd = macro_data.get("usdinr") or {}
    fx_week = abs(float(usd.get("week_change", 0) or 0))
    if fx_week >= PRIORITIZER_FX_WEEK_ANCHOR:
        push(
            "synthetic_fx_anchor",
            f"DATA-ANCHOR: USD/INR week_change {usd.get('week_change')} ({usd.get('direction', '')})",
            "fxStress",
            None,
        )

    return out


def _apply_hybrid_tiebreak(selected: list[dict], memory_summary: dict) -> tuple[list[dict], list[dict]]:
    if not selected:
        return selected, []
    payload = {"shortlist": selected[:8], "memory_summary": memory_summary}
    try:
        response = prioritizer_llm.invoke([
            SystemMessage(content=PRIORITIZER_TIEBREAKER_PROMPT),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        text = response.content if hasattr(response, "content") else str(response)
        parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
        ranked = parsed.get("ranked", [])
        by_id = {s.get("story_id"): s for s in selected}
        out: list[dict] = []
        deltas: list[dict] = []
        for idx, item in enumerate(ranked, start=1):
            sid = item.get("story_id")
            if sid in by_id:
                base = dict(by_id[sid])
                base["tie_break_meta"] = {
                    "rank": item.get("rank", idx),
                    "why_now": item.get("why_now", ""),
                    "contrarian_angle": item.get("contrarian_angle", ""),
                    "confidence": item.get("confidence", "medium"),
                }
                out.append(base)
        if out:
            out = out[:PRIORITIZER_TOP_K]
            prev_order = {s.get("story_id"): i + 1 for i, s in enumerate(selected)}
            for i, s in enumerate(out, start=1):
                sid = s.get("story_id")
                if sid in prev_order and prev_order[sid] != i:
                    deltas.append({"story_id": sid, "from": prev_order[sid], "to": i})
            return out, deltas
    except Exception:
        pass
    return selected, []


def news_prioritizer_node(state: AgentState) -> dict:
    news_articles = state.get("news_articles", [])
    market_data = state.get("market_data", {})
    macro_data = state.get("macro_data", {})

    memory = _load_prioritizer_memory()
    memory_summary = _memory_summary(memory)
    recent_fingerprints = set(memory_summary.get("recent_fingerprints", []))

    candidates: list[dict] = []
    novelty_penalties: list[dict] = []
    for idx, article in enumerate(news_articles):
        headline = article.get("title", "")
        summary = article.get("summary", "")
        if not headline:
            continue
        theme = _infer_story_theme(article, market_data, macro_data)
        fingerprint = hashlib.md5(_normalize_headline(headline).encode("utf-8")).hexdigest()[:12]
        candidate = {
            "story_id": f"story_{idx+1}_{fingerprint}",
            "headline": headline,
            "summary": summary,
            "theme": theme,
            "published": article.get("published", ""),
            "sources": [{
                "title": headline,
                "url": article.get("url", ""),
                "provider": article.get("provider", ""),
                "source": article.get("source", ""),
                "published": article.get("published", ""),
            }],
            "hard_facts": _extract_hard_facts_for_theme(theme, article, market_data, macro_data),
            "transmission_path": _build_transmission_path(theme, market_data, macro_data),
            "confidence": "medium",
            "fingerprint": fingerprint,
            "synthetic_anchor": False,
        }
        score = _score_story_candidate(candidate, market_data, macro_data, memory_summary)
        candidate["score_total"] = score["total"]
        candidate["score_breakdown"] = score["breakdown"]
        if fingerprint in recent_fingerprints:
            candidate["score_total"] = max(0, candidate["score_total"] - 1)
            candidate["score_breakdown"]["novelty_proxy"] = 0
            novelty_penalties.append({"story_id": candidate["story_id"], "reason": "recent_repeat"})
        candidates.append(candidate)

    candidates.extend(_build_synthetic_macro_candidates(market_data, macro_data))
    deduped = _dedupe_story_candidates(candidates)
    ranked = sorted(deduped, key=lambda c: c.get("score_total", 0), reverse=True)
    selected, guardrail_meta = _select_diverse_stories(ranked)
    tie_break_deltas: list[dict] = []
    tie_break_applied = False
    if PRIORITIZER_MODE == "hybrid":
        selected, tie_break_deltas = _apply_hybrid_tiebreak(selected, memory_summary)
        tie_break_applied = True if tie_break_deltas or selected else False

    selected_theme_mix = dict(Counter([s.get("theme", "unknown") for s in selected]))

    new_run = {
        "ts": datetime.utcnow().isoformat(),
        "themes": [s.get("theme", "unknown") for s in selected],
        "story_ids": [s.get("story_id", "") for s in selected],
        "fingerprints": [s.get("fingerprint", "") for s in selected if s.get("fingerprint")],
    }
    memory_runs = (memory.get("runs", []) + [new_run])[-PRIORITIZER_MEMORY_WINDOW:]
    memory["runs"] = memory_runs
    _save_prioritizer_memory(memory)

    return {
        "story_candidates": ranked,
        "selected_stories": selected,
        "prioritizer_memory": memory_summary,
        "coverage_telemetry": {
            "prioritizer_mode": PRIORITIZER_MODE,
            "selected_count": len(selected),
            "selected_theme_mix": selected_theme_mix,
            "rejected_summary": guardrail_meta.get("rejected_summary", {}),
            "novelty_penalties": novelty_penalties,
            "tie_break_applied": tie_break_applied,
            "tie_break_deltas": tie_break_deltas,
        },
    }

# ---------------------------------------------------------------------------
# Context fetch + Writer node
# ---------------------------------------------------------------------------

writer_tools = [get_market_data, get_macro_data, fetch_news]

writer_agent = create_agent(
    writer_llm,
    writer_tools,
    system_prompt=WRITER_PROMPT,
)


def context_fetch_node(state: AgentState) -> dict:
    """Fetch market, macro, and news deterministically once per run."""
    market_data = state.get("market_data") or {}
    macro_data = state.get("macro_data") or {}
    news_articles = state.get("news_articles") or []

    if not market_data:
        try:
            market_data = get_market_data.invoke({})
        except Exception:
            market_data = {}
    if not macro_data:
        try:
            macro_data = get_macro_data.invoke({})
        except Exception:
            macro_data = {}
    if not news_articles:
        try:
            news_articles = fetch_news.invoke({})
        except Exception:
            news_articles = []

    return {
        "market_data": market_data,
        "macro_data": macro_data,
        "news_articles": news_articles,
    }


def writer_node(state: AgentState) -> dict:
    """
    Generate a batch of tweets.

    Batch 1: calls tools to fetch market data + macro data + news, then writes tweets.
    Later batches: uses existing data, knows how many approved tweets exist,
    and generates NEW tweets with different angles.
    """
    attempt = state.get("attempt", 0) + 1
    approved = state.get("approved_tweets", [])
    selected_stories = state.get("selected_stories", [])

    selected_block = (
        "SELECTED_STORIES (primary candidates):\n"
        + json.dumps(selected_stories, ensure_ascii=False, indent=2)
        + "\n\nUse these shortlisted stories first. Only use raw news for corroboration."
    )
    channel_block = WRITER_TWEET_CHANNEL_GUIDE

    if attempt == 1:
        result = writer_agent.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "Analyze today's market and macro context, then write tweets.\n\n"
                            f"{channel_block}\n\n{selected_block}"
                        )
                    )
                ]
            }
        )
    else:
        batch_msg = WRITER_BATCH_TEMPLATE.format(
            approved_count=len(approved),
            remaining=TOTAL_NEEDED - len(approved),
            TOTAL_NEEDED=TOTAL_NEEDED,
            feedback=state.get("audit_feedback", ""),
            batch_size=TWEET_BATCH_SIZE,
        )
        result = writer_agent.invoke(
            {
                "messages": [
                    HumanMessage(content=f"{batch_msg}\n\n{channel_block}\n\n{selected_block}"),
                ]
            }
        )

    last_message = result["messages"][-1]
    content = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
    tweets = _parse_tweets(content)
    tweets = _validate_tweets_structure(tweets)

    market_data = state.get("market_data", {})
    macro_data = state.get("macro_data", {})
    news_articles = state.get("news_articles", [])

    if attempt == 1:
        for msg in result["messages"]:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                try:
                    parsed = json.loads(msg.content)
                    if isinstance(parsed, dict) and "indices" in parsed:
                        market_data = parsed
                    elif isinstance(parsed, dict) and "rbi_rate" in parsed:
                        macro_data = parsed
                    elif isinstance(parsed, list) and len(parsed) > 0 and "title" in parsed[0]:
                        news_articles = parsed
                except (json.JSONDecodeError, TypeError, IndexError):
                    pass

    return {
        "draft_tweets": tweets,
        "market_data": market_data,
        "macro_data": macro_data,
        "news_articles": news_articles,
        "attempt": attempt,
    }

def _parse_tweets(content: str) -> list[str]:
    """Extract tweet list from LLM response."""
    try:
        start = content.index("[")
        end = content.rindex("]") + 1
        tweets = json.loads(content[start:end])
        if isinstance(tweets, list) and all(isinstance(t, str) for t in tweets):
            return tweets
    except (ValueError, json.JSONDecodeError):
        pass

    lines = [
        line.strip().lstrip("0123456789.-) ").strip('"')
        for line in content.split("\n")
        if line.strip() and len(line.strip()) > 20
    ]
    return lines[:TWEET_BATCH_SIZE] if lines else [content.strip()]


def _validate_tweets_structure(tweets: list[str]) -> list[str]:
    """
    Deterministically enforce the strict 4-part Writer structure.

    The prompts instruct the Writer to output each tweet as a SINGLE-LINE string
    containing these labels:
      - PUNCHLINE_HEADLINE:
      - LEAD_PARAGRAPH:
      - MECHANICAL_CONTRARIAN_EXPLANATION:
      - INDEX_SNAPSHOT_GLOBAL_CONTEXT:
    """
    required_labels = [
        "PUNCHLINE_HEADLINE:",
        "LEAD_PARAGRAPH:",
        "MECHANICAL_CONTRARIAN_EXPLANATION:",
        "INDEX_SNAPSHOT_GLOBAL_CONTEXT:",
    ]

    def _sanitize_single_line(s: str) -> str:
        # Remove actual newlines that would break JSON-safe "single line" output.
        return s.replace("\r", " ").replace("\n", " ").strip()

    tweets = [_sanitize_single_line(str(t)) for t in (tweets or [])]

    validated: list[str] = []
    for t in tweets:
        missing = [lbl for lbl in required_labels if lbl not in t]
        if missing:
            marker = f"STRUCTURE_INVALID_MISSING_LABELS[{','.join(missing)}]"
            validated.append(f"{marker} {t}".strip())
        else:
            validated.append(t)

    # Keep downstream assumptions stable: always return exactly TWEET_BATCH_SIZE items.
    if len(validated) < TWEET_BATCH_SIZE:
        while len(validated) < TWEET_BATCH_SIZE:
            validated.append("STRUCTURE_INVALID_TOO_FEW_TWEETS")
    elif len(validated) > TWEET_BATCH_SIZE:
        validated = validated[:TWEET_BATCH_SIZE]

    return validated

# ---------------------------------------------------------------------------
# Auditor node
# ---------------------------------------------------------------------------

def auditor_node(state: AgentState) -> dict:
    """
    Score each tweet in the current batch.
    Tweets scoring >= AUDIT_PASS_THRESHOLD are added to the accumulator.
    """
    drafts = state.get("draft_tweets", [])
    market_data = state.get("market_data", {})
    macro_data = state.get("macro_data", {})
    news_articles = state.get("news_articles", [])
    approved = list(state.get("approved_tweets", []))

    audit_input = (
        f"TWEET DRAFTS:\n{json.dumps(drafts, indent=2)}\n\n"
        f"MARKET DATA:\n{json.dumps(market_data, indent=2)}\n\n"
        f"MACRO DATA:\n{json.dumps(macro_data, indent=2)}\n\n"
        f"NEWS ARTICLES:\n{json.dumps(news_articles, indent=2)}"
    )

    try:
        response = auditor_llm.invoke([
            SystemMessage(content=AUDITOR_PROMPT),
            HumanMessage(content=audit_input),
        ])
    except Exception:
        return {
            "scored_tweets": [
                {"tweet": t, "score": 7, "breakdown": {}, "feedback": "Auditor rate-limited, scored conservatively"}
                for t in drafts
            ],
            "approved_tweets": approved,
            "audit_feedback": "Auditor was rate-limited. Tweets scored conservatively at 7/12.",
        }

    scored_tweets = _parse_audit_response(response.content, drafts)

    # Accumulate approved tweets (score >= threshold)
    for tweet in scored_tweets:
        if tweet.get("score", 0) >= AUDIT_PASS_THRESHOLD:
            approved.append(tweet)

    # Build feedback for writer (only from rejected tweets in this batch)
    rejected = [t for t in scored_tweets if t.get("score", 0) < AUDIT_PASS_THRESHOLD]
    feedback_lines = [
        f"Score {t['score']}/12: \"{t['tweet'][:80]}...\" -- {t['feedback']}"
        for t in rejected
    ]
    feedback = "\n".join(feedback_lines) if feedback_lines else "All tweets in this batch were approved."

    return {
        "scored_tweets": scored_tweets,
        "approved_tweets": approved,
        "audit_feedback": feedback,
    }

def _parse_audit_response(content: str, original_drafts: list[str]) -> list[dict]:
    """Parse the auditor's JSON response."""
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        data = json.loads(content[start:end])
        tweets = data.get("tweets", [])
        if tweets:
            return tweets
    except (ValueError, json.JSONDecodeError):
        pass

    return [
        {"tweet": t, "score": 5, "breakdown": {}, "feedback": "Audit parse failed"}
        for t in original_drafts
    ]

# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_audit(state: AgentState) -> str:
    """
    Decide next step:
    - "synthesize"  -> 5+ approved tweets, send to synthesizer
    - "force_synth" -> max batches hit, synthesize best available
    - "next_batch"  -> need more good tweets, generate another batch
    """
    approved = state.get("approved_tweets", [])
    attempt = state.get("attempt", 1)

    if len(approved) >= TOTAL_NEEDED:
        return "synthesize"

    if attempt >= MAX_AUDIT_CYCLES:
        return "force_synth"

    return "next_batch"

# ---------------------------------------------------------------------------
# Synthesizer node -- combines approved tweets into premium output
# ---------------------------------------------------------------------------

def synthesizer_node(state: AgentState) -> dict:
    """
    Combine the best approved tweets into:
    1. One premium thread (1000-1500 chars) with 3-act narrative arc
    2. One standalone punchy tweet (under 280 chars)
    """
    approved = state.get("approved_tweets", [])
    market_data = state.get("market_data", {})
    macro_data = state.get("macro_data", {})
    news_articles = state.get("news_articles", [])
    claim_evidence = _build_claim_evidence_graph(market_data, macro_data, news_articles)
    regime = _classify_regime(market_data, macro_data)
    coverage_telemetry = state.get("coverage_telemetry", {})

    top_tweets = sorted(approved, key=lambda t: t.get("score", 0), reverse=True)
    top_tweets = top_tweets[:TOTAL_NEEDED]
    selected_stories = state.get("selected_stories", [])

    synth_input = (
        f"APPROVED TWEETS (your raw material -- combine the best elements):\n"
        + "\n\n".join(
            f"[Score {t.get('score', '?')}/12] {t.get('tweet', '')}"
            for t in top_tweets
        )
        + f"\n\nSELECTED_STORIES (editorial shortlist -- weave commodity vs India tape if both appear):\n"
        + json.dumps(selected_stories, ensure_ascii=False, indent=2)
        + f"\n\nREGIME:\n{regime}"
        + f"\n\nCLAIM_EVIDENCE_GRAPH:\n{json.dumps(claim_evidence, indent=2)}"
        + f"\n\nMARKET DATA:\n{json.dumps(market_data, indent=2)}"
        + f"\n\nMACRO DATA:\n{json.dumps(macro_data, indent=2)}"
        + f"\n\nNEWS ARTICLES:\n{json.dumps(news_articles[:10], indent=2)}"
    )

    response = synthesizer_llm.invoke([
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=synth_input),
    ])

    final = _parse_synthesis(response.content)
    final = _sanitize_dispatch_output(final)
    qa = _validate_dispatch_output(
        final,
        regime=regime,
        claim_evidence=claim_evidence,
        coverage_telemetry=coverage_telemetry,
    )

    rewrite_attempted = False
    rewrite_passed = False
    if not qa["passed"]:
        rewrite_attempted = True
        repaired = _repair_dispatch_once(final, qa.get("failed_rules", []), synth_input)
        repaired = _sanitize_dispatch_output(repaired)
        repaired_qa = _validate_dispatch_output(
            repaired,
            regime=regime,
            claim_evidence=claim_evidence,
            coverage_telemetry=coverage_telemetry,
        )
        if repaired_qa["passed"]:
            final = repaired
            qa = repaired_qa
            rewrite_passed = True
        else:
            qa = repaired_qa

    return {
        "final_tweets": final,
        "approved_tweets": top_tweets,
        "dispatch_qa": {
            "dispatch_validation_passed": qa["passed"],
            "failed_rules": qa.get("failed_rules", []),
            "hard_fail_reasons": qa.get("hard_fail_reasons", []),
            "qa_v2_scores": qa.get("qa_v2_scores", {}),
            "weighted_total": qa.get("weighted_total"),
            "rewrite_attempted": rewrite_attempted,
            "rewrite_passed": rewrite_passed,
            "rewrite_reason_codes": qa.get("failed_rules", []),
            "coverage_telemetry": coverage_telemetry,
        },
        "claim_evidence": claim_evidence,
        "regime": regime,
    }

def _coerce_message_content_to_text(message_or_content) -> str:
    """Normalize LangChain AIMessage / str / content blocks to a single string."""
    content = getattr(message_or_content, "content", message_or_content)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                t = block.get("text")
                if t is not None:
                    parts.append(str(t))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)


def _strip_markdown_code_fence(text: str) -> str:
    t = (text or "").strip()
    if not t.startswith("```"):
        return t
    lines = t.split("\n")
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_json_object_slice(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _json_loads_object(blob: str) -> dict | None:
    blob = (blob or "").strip()
    if not blob:
        return None
    blob = blob.replace("\u201c", '"').replace("\u201d", '"')
    try:
        data = json.loads(blob)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        fixed = re.sub(r",(\s*[}\]])", r"\1", blob)
        data = json.loads(fixed)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _unescape_json_string_fragments(s: str) -> str:
    """Undo common JSON escapes in a substring extracted without full json.loads."""
    out: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "t":
                out.append("\t")
                i += 2
                continue
            if nxt in "\"\\/":
                out.append(nxt)
                i += 2
                continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _find_closing_quote_index(s: str) -> int | None:
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            i += 2
            continue
        if s[i] == '"':
            return i
        i += 1
    return None


def _loose_extract_synthesis_fields(text: str) -> dict[str, str] | None:
    """
    Extract premium_thread / standalone when the model emits pseudo-JSON (e.g. raw
    newlines inside string values), which makes json.loads fail.
    """
    if not text:
        return None
    for sa_key in ("standalone_tweet", "standalone"):
        m = re.search(r'"premium_thread"\s*:\s*"', text, re.IGNORECASE | re.DOTALL)
        if not m:
            m = re.search(r"'premium_thread'\s*:\s*'", text, re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        val0 = m.end()
        sep = re.compile(
            rf'"\s*,\s*"{re.escape(sa_key)}"\s*:\s*"',
            re.IGNORECASE | re.DOTALL,
        )
        sm = sep.search(text, val0)
        if not sm:
            continue
        premium_raw = text[val0 : sm.start()]
        sa_val_start = sm.end()
        end_sq = _find_closing_quote_index(text[sa_val_start:])
        if end_sq is None:
            continue
        standalone_raw = text[sa_val_start : sa_val_start + end_sq]
        return {
            "premium_thread": _unescape_json_string_fragments(premium_raw),
            "standalone_tweet": _unescape_json_string_fragments(standalone_raw),
        }
    return None


def _synthesis_dict_to_items(data: dict) -> list[dict]:
    result: list[dict] = []
    pt = data.get("premium_thread") or data.get("premiumThread")
    sa = (
        data.get("standalone_tweet")
        or data.get("standaloneTweet")
        or data.get("standalone")
    )
    if isinstance(pt, str) and pt.strip():
        result.append({"tweet": pt.strip(), "type": "premium_thread", "score": 12})
    if isinstance(sa, str) and sa.strip():
        result.append({"tweet": sa.strip(), "type": "standalone", "score": 12})
    return result


def _parse_synthesis(content) -> list[dict]:
    """Parse synthesizer JSON response into final tweets list (OpenAI or Ollama)."""
    text = _coerce_message_content_to_text(content)
    text = _strip_markdown_code_fence(text)
    if not text.strip():
        return [{"tweet": "", "type": "raw", "score": 0}]

    blob = _extract_json_object_slice(text) or text.strip()
    data = _json_loads_object(blob)
    if data is None:
        data = _loose_extract_synthesis_fields(blob)
    if data is None:
        data = _loose_extract_synthesis_fields(text)

    if isinstance(data, dict):
        items = _synthesis_dict_to_items(data)
        if items:
            return items

    return [{"tweet": text.strip(), "type": "raw", "score": 0}]


def _sanitize_dispatch_output(final_tweets: list[dict]) -> list[dict]:
    """Deterministically enforce the Premium Institutional Dispatch format."""
    sanitized = []
    for item in final_tweets or []:
        if item.get("type") == "premium_thread" and isinstance(item.get("tweet"), str):
            item = dict(item)
            item["tweet"] = _sanitize_premium_dispatch_thread(item["tweet"])
        sanitized.append(item)
    return sanitized


def _validate_dispatch_output(
    final_tweets: list[dict],
    regime: str = "",
    claim_evidence: dict | None = None,
    coverage_telemetry: dict | None = None,
) -> dict:
    """Run score-first QA v2 with explicit hard-fail gates."""
    premium = next((t for t in (final_tweets or []) if t.get("type") == "premium_thread"), None)
    if not premium or not isinstance(premium.get("tweet"), str):
        return {
            "passed": False,
            "failed_rules": ["missing_premium_thread"],
            "hard_fail_reasons": ["missing_premium_thread"],
            "qa_v2_scores": {},
        }

    text = premium["tweet"]
    failed = []
    hard_fail_reasons = []
    scores = {
        "regimeAlignment": 2,
        "evidenceDiscipline": 2,
        "causalClarity": 2,
        "mechanicalDepth": 2,
        "institutionalVoice": 2,
        "watchFrameworkQuality": 2,
        "topicDiversity": 2,
        "crossAssetDepth": 2,
        "indiaTransmissionClarity": 2,
        "noveltyVsRecentRuns": 2,
        "redundancyPenalty": 0,
    }

    # Hard fail 1: headline presence.
    if not text.startswith("HEADLINE:"):
        hard_fail_reasons.append("missing_headline_prefix")
        failed.append("missing_headline_prefix")
        scores["institutionalVoice"] = max(0, scores["institutionalVoice"] - 1)

    lowered = text.lower()
    banned_phrases = [
        "[act",
        "act 1",
        "act 2",
        "act 3",
        "summary",
        "conflict",
        "in conclusion",
        "the real question is",
        "it's interesting to note",
    ]
    for phrase in banned_phrases:
        if phrase in lowered:
            failed.append(f"banned_phrase:{phrase}")
            hard_fail_reasons.append(f"banned_phrase:{phrase}")
            scores["institutionalVoice"] = max(0, scores["institutionalVoice"] - 1)

    parts = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Expect exactly: HEADLINE + 3 body paragraphs
    if len(parts) != 4:
        failed.append("paragraph_count_not_exactly_3_body")
        scores["causalClarity"] = max(0, scores["causalClarity"] - 1)

    # Metric density: require at least 6 numeric tokens.
    num_tokens = re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?", text)
    if len(num_tokens) < 6:
        failed.append("low_metric_density")
        scores["evidenceDiscipline"] = max(0, scores["evidenceDiscipline"] - 1)
        scores["crossAssetDepth"] = max(0, scores["crossAssetDepth"] - 1)

    institutional_tokens = [
        "institutional",
        "positioning unwind",
        "de-risking",
        "risk aversion",
        "systematic hedge funds",
        "liquidity",
        "rotation",
        "margin",
        "forced selling",
    ]
    if not any(tok in lowered for tok in institutional_tokens):
        failed.append("missing_institutional_framing")
        scores["institutionalVoice"] = max(0, scores["institutionalVoice"] - 1)

    transition_markers = [
        "this price action underscores",
        "parallel to this move",
        "the divergence suggests",
    ]
    if not any(tok in lowered for tok in transition_markers):
        failed.append("missing_transition_connector")
        scores["causalClarity"] = max(0, scores["causalClarity"] - 1)

    # Redundancy penalty: repeated paragraph openings/phrases.
    paragraphs = [p.strip().lower() for p in parts[1:]]
    if len(paragraphs) >= 2:
        starts = [p[:80] for p in paragraphs]
        if len(starts) != len(set(starts)):
            failed.append("redundant_paragraph_openings")
            scores["redundancyPenalty"] += 1
            scores["mechanicalDepth"] = max(0, scores["mechanicalDepth"] - 1)

    # Hard fail 2: final paragraph must include exact watch-grid fields.
    final_paragraph = paragraphs[-1] if paragraphs else ""
    has_watch_grid = (
        "trigger:" in final_paragraph
        and "confirmation:" in final_paragraph
        and "invalidation:" in final_paragraph
    )
    if not has_watch_grid:
        failed.append("missing_watch_grid")
        hard_fail_reasons.append("missing_watch_grid")
        scores["watchFrameworkQuality"] = max(0, scores["watchFrameworkQuality"] - 1)

    india_terms = ["india", "nifty", "sensex", "usd/inr", "rupee", "fii", "dii"]
    if not any(tok in lowered for tok in india_terms):
        failed.append("weak_india_transmission")
        scores["indiaTransmissionClarity"] = max(0, scores["indiaTransmissionClarity"] - 1)

    # Regime alignment heuristic (dimension-specific).
    regime_markers = {
        "riskOffUnwind": ["risk aversion", "forced selling", "margin", "liquidity", "unwind"],
        "commodityShock": ["input cost", "supply", "cost-push", "crude", "brent"],
        "policyRepricing": ["yield", "dxy", "carry trade", "central bank", "policy"],
        "idiosyncraticRotation": ["rotation", "sector", "dispersion", "reallocation"],
    }
    markers = regime_markers.get(regime, [])
    if markers and not any(m in lowered for m in markers):
        scores["regimeAlignment"] = max(0, scores["regimeAlignment"] - 1)
        failed.append("weak_regime_alignment")

    # Evidence discipline: check at least one hard_fact claim anchor appears.
    if claim_evidence:
        hard_claims = (claim_evidence.get("hard_fact") or [])[:4]
        anchor_found = False
        for hc in hard_claims:
            claim_text = str(hc.get("claim", ""))
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", claim_text)
            # If at least one numeric anchor from hard facts appears in output, treat as grounded.
            if any(n in text for n in nums):
                anchor_found = True
                break
        if not anchor_found:
            scores["evidenceDiscipline"] = max(0, scores["evidenceDiscipline"] - 1)
            failed.append("weak_evidence_grounding")

    if coverage_telemetry:
        selected_theme_mix = coverage_telemetry.get("selected_theme_mix", {})
        if len(selected_theme_mix) < 2:
            failed.append("low_topic_diversity")
            scores["topicDiversity"] = max(0, scores["topicDiversity"] - 1)
        novelty_penalties = coverage_telemetry.get("novelty_penalties", [])
        if novelty_penalties:
            scores["noveltyVsRecentRuns"] = max(0, scores["noveltyVsRecentRuns"] - 1)

    # Score-first: pass if no hard fails and weighted score above threshold.
    weighted_total = (
        scores["regimeAlignment"]
        + scores["evidenceDiscipline"]
        + scores["causalClarity"]
        + scores["mechanicalDepth"]
        + scores["institutionalVoice"]
        + scores["watchFrameworkQuality"]
        + scores["topicDiversity"]
        + scores["crossAssetDepth"]
        + scores["indiaTransmissionClarity"]
        + scores["noveltyVsRecentRuns"]
        - scores["redundancyPenalty"]
    )
    passed = len(hard_fail_reasons) == 0 and weighted_total >= 8

    return {
        "passed": passed,
        "failed_rules": failed,
        "hard_fail_reasons": hard_fail_reasons,
        "qa_v2_scores": scores,
        "weighted_total": weighted_total,
    }


def _repair_dispatch_once(final_tweets: list[dict], failed_rules: list[str], synth_input: str) -> list[dict]:
    """One strict rewrite pass if deterministic dispatch checks fail."""
    premium = next((t for t in (final_tweets or []) if t.get("type") == "premium_thread"), None)
    standalone = next((t for t in (final_tweets or []) if t.get("type") == "standalone"), None)
    raw = next((t for t in (final_tweets or []) if t.get("type") == "raw"), None)

    if (not premium or not isinstance(premium.get("tweet"), str)) and raw and isinstance(
        raw.get("tweet"), str
    ):
        recovered = _parse_synthesis(raw["tweet"])
        if recovered and all(t.get("type") != "raw" for t in recovered):
            final_tweets = list(recovered)
            premium = next((t for t in final_tweets if t.get("type") == "premium_thread"), None)
            standalone = next((t for t in final_tweets if t.get("type") == "standalone"), None)

    if not premium or not isinstance(premium.get("tweet"), str):
        return final_tweets

    repair_system = (
        "You are fixing a premium market dispatch using failed QA dimensions only.\n"
        "Return ONLY JSON with keys premium_thread and standalone_tweet.\n"
        "Do NOT rewrite everything. Preserve valid content and repair only failed dimensions.\n"
        "Rules:\n"
        "- premium_thread MUST start with 'HEADLINE:'\n"
        "- Exactly 3 body paragraphs after headline\n"
        "- Paragraph 3 MUST contain Trigger:/Confirmation:/Invalidation:\n"
        "- No scaffolding tokens: [ACT, ACT 1/2/3, Summary, Conflict, In conclusion\n"
        "- Use regime and claim-evidence context from SOURCE_CONTEXT\n"
        "- Treat weak inference as speculative chatter only\n"
        "- Plain text only\n"
    )
    if LLM_PROVIDER == "ollama":
        repair_system = repair_system + "\n" + OLLAMA_SYNTHESIZER_JSON_SUFFIX
    repair_user = (
        "FAILED_RULES:\n"
        + json.dumps(failed_rules, indent=2)
        + "\n\nORIGINAL_PREMIUM_THREAD:\n"
        + premium["tweet"]
        + "\n\nORIGINAL_STANDALONE:\n"
        + (standalone.get("tweet", "") if standalone else "")
        + "\n\nFAILED_DIMENSIONS_ONLY_FIX:\n"
        + ", ".join(failed_rules)
        + "\n\nSOURCE_CONTEXT:\n"
        + synth_input
    )

    try:
        response = synthesizer_llm.invoke([
            SystemMessage(content=repair_system),
            HumanMessage(content=repair_user),
        ])
        repaired = _parse_synthesis(_coerce_message_content_to_text(response))
        return repaired if repaired else final_tweets
    except Exception:
        return final_tweets


def _sanitize_premium_dispatch_thread(thread: str) -> str:
    """
    Minimal fallback-only sanitizer:
    - ensure HEADLINE exists
    - strip forbidden scaffolding tokens
    - preserve model text as much as possible
    """
    t = (thread or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    # Strip forbidden scaffolding markers only.
    t = re.sub(r"\[ACT[^\]]*\]", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bACT\s*[123]\b\s*[-:]*\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\b(Summary|Conflict|In conclusion)\b\s*[:\-]*", "", t, flags=re.IGNORECASE)
    t = t.strip()

    # Guarantee HEADLINE.
    if "HEADLINE:" not in t:
        first = t.split("\n", 1)[0].strip() if t else "MARKET UPDATE"
        if not first:
            first = "MARKET UPDATE"
        # Keep title-case style for Institutional Narrative Engine.
        headline = " ".join(word.capitalize() for word in first[:120].split())
        t = f"HEADLINE: {headline}\n\n{t}"

    # Keep text unchanged beyond minimal cleanup.
    return t


def _build_claim_evidence_graph(market_data: dict, macro_data: dict, news_articles: list[dict]) -> dict:
    """Build deterministic claim-evidence graph for synthesis conditioning."""
    graph = {"hard_fact": [], "mechanical_inference": [], "weak_inference": []}

    indices = market_data.get("indices", {})
    for name, data in indices.items():
        if isinstance(data, dict) and "price" in data and "change_pct" in data:
            graph["hard_fact"].append(
                {
                    "claim": f"{name} moved {data['change_pct']:+.2f}% to {data['price']}",
                    "evidence": {"source": "market_data.indices", "name": name, "data": data},
                }
            )

    vix = market_data.get("india_vix")
    if isinstance(vix, dict) and vix.get("value") is not None:
        graph["hard_fact"].append(
            {
                "claim": f"India VIX at {vix.get('value')} vs 5d avg {vix.get('avg_5d')}",
                "evidence": {"source": "market_data.india_vix", "data": vix},
            }
        )

    usdinr = macro_data.get("usdinr")
    if isinstance(usdinr, dict) and usdinr.get("spot_rate") is not None:
        graph["hard_fact"].append(
            {
                "claim": f"USD/INR at {usdinr.get('spot_rate')} with week change {usdinr.get('week_change')}",
                "evidence": {"source": "macro_data.usdinr", "data": usdinr},
            }
        )

    movers = market_data.get("top_movers", {})
    breadth = movers.get("breadth", {})
    if isinstance(breadth, dict) and breadth.get("green") is not None:
        graph["mechanical_inference"].append(
            {
                "claim": "Weak breadth implies risk reduction and selective liquidity preference.",
                "evidence": {"source": "market_data.top_movers.breadth", "data": breadth},
            }
        )

    if isinstance(vix, dict) and vix.get("change_pct") is not None and float(vix.get("change_pct", 0)) > 10:
        graph["mechanical_inference"].append(
            {
                "claim": "Elevated volatility indicates institutional hedging demand and de-risking bias.",
                "evidence": {"source": "market_data.india_vix", "data": vix},
            }
        )

    if news_articles:
        sample_news = [{"title": n.get("title", ""), "source": n.get("source", "")} for n in news_articles[:5]]
        graph["weak_inference"].append(
            {
                "claim": "Headline flow points to macro uncertainty, but causal hierarchy requires market confirmation.",
                "evidence": {"source": "news_articles[:5]", "data": sample_news},
            }
        )

    return graph


def _classify_regime(market_data: dict, macro_data: dict) -> str:
    """Classify deterministic market regime from available metrics."""
    vix = market_data.get("india_vix") or {}
    breadth = ((market_data.get("top_movers") or {}).get("breadth") or {})
    global_data = market_data.get("global") or {}
    crude = global_data.get("Crude Oil") or {}
    brent = global_data.get("Brent") or {}
    usdinr = macro_data.get("usdinr") or {}

    vix_jump = float(vix.get("change_pct", 0) or 0)
    green = int(breadth.get("green", 0) or 0)
    total = int(breadth.get("total", 0) or 0)
    breadth_weak = total > 0 and green / total < 0.3
    crude_move = abs(float(crude.get("change_pct", 0) or 0))
    brent_move = abs(float(brent.get("change_pct", 0) or 0))
    fx_week = abs(float(usdinr.get("week_change", 0) or 0))

    if vix_jump >= 10 and breadth_weak:
        return "riskOffUnwind"
    if crude_move >= 2 or brent_move >= 2:
        return "commodityShock"
    if fx_week >= 0.6:
        return "policyRepricing"
    return "idiosyncraticRotation"

# ---------------------------------------------------------------------------
# Deliver node
# ---------------------------------------------------------------------------

def deliver_node(state: AgentState) -> dict:
    """Send synthesized tweets + individual approved tweets via email (or skip when EMAIL_DELIVERY=terminal)."""
    final_tweets = state.get("final_tweets", [])
    approved = state.get("approved_tweets", [])
    market_data = state.get("market_data", {})
    macro_data = state.get("macro_data", {})
    news_articles = state.get("news_articles", [])
    dispatch_qa = state.get("dispatch_qa", {})

    if EMAIL_DELIVERY == "terminal":
        return {
            "audit_feedback": "Email skipped (EMAIL_DELIVERY=terminal)",
            "dispatch_qa": dispatch_qa,
        }

    all_tweets = final_tweets + approved

    result = send_email.invoke({
        "scored_tweets": all_tweets,
        "market_data": market_data,
        "macro_data": macro_data,
        "news_articles": news_articles,
        "dispatch_qa": dispatch_qa,
    })

    return {
        "audit_feedback": f"Email result: {result}",
        "dispatch_qa": dispatch_qa,
    }

# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    """
    Build and compile the LangGraph.

    Flow:
      writer -> auditor -> (conditional)
        -> synthesizer -> deliver  (if 5+ approved)
        -> synthesizer -> deliver  (if max batches hit)
        -> writer                  (if need more good tweets)
    """
    graph = StateGraph(AgentState)

    graph.add_node("context_fetch", context_fetch_node)
    graph.add_node("news_prioritizer", news_prioritizer_node)
    graph.add_node("writer", writer_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("deliver", deliver_node)

    graph.set_entry_point("context_fetch")

    graph.add_edge("context_fetch", "news_prioritizer")
    graph.add_edge("news_prioritizer", "writer")
    graph.add_edge("writer", "auditor")

    graph.add_conditional_edges(
        "auditor",
        route_after_audit,
        {
            "synthesize": "synthesizer",
            "force_synth": "synthesizer",
            "next_batch": "writer",
        },
    )

    graph.add_edge("synthesizer", "deliver")
    graph.add_edge("deliver", END)

    return graph.compile()