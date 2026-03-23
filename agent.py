"""
LangGraph agent -- the brain of the tweet bot.

Batch accumulation graph:
  Writer generates 3 tweets/batch -> Auditor scores them ->
  approved tweets (>= 8) accumulate -> loop until 5 good tweets collected.
"""

import json
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import END, StateGraph

from config import (
    AUDIT_PASS_THRESHOLD,
    AUDITOR_MODEL,
    MAX_AUDIT_CYCLES,
    OPENAI_API_KEY,
    TOTAL_NEEDED,
    TWEET_BATCH_SIZE,
    WRITER_MODEL,
)
from prompts import AUDITOR_PROMPT, SYNTHESIZER_PROMPT, WRITER_BATCH_TEMPLATE, WRITER_PROMPT
from tools.email import send_email
from tools.market import get_market_data
from tools.news import fetch_news


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    market_data: dict
    news_articles: list[dict]
    draft_tweets: list[str]          # Current batch (3 tweets)
    scored_tweets: list[dict]        # Current batch scored
    approved_tweets: list[dict]      # Accumulator -- good tweets across ALL batches
    final_tweets: list[dict]         # Synthesizer output (premium thread + standalone)
    audit_feedback: str
    attempt: int                     # Batch number (1-based)


# ---------------------------------------------------------------------------
# LLMs -- OpenAI for both Writer and Auditor
# ---------------------------------------------------------------------------

writer_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=WRITER_MODEL,
    temperature=0.7,
)

auditor_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=AUDITOR_MODEL,
    temperature=0.1,
)

synthesizer_llm = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model=WRITER_MODEL,
    temperature=0.4,
)


# ---------------------------------------------------------------------------
# Writer node
# ---------------------------------------------------------------------------

writer_tools = [get_market_data, fetch_news]

writer_agent = create_agent(
    writer_llm,
    writer_tools,
    system_prompt=WRITER_PROMPT,
)


def writer_node(state: AgentState) -> dict:
    """
    Generate a batch of tweets.

    Batch 1: calls tools to fetch market data + news, then writes tweets.
    Later batches: uses existing data, knows how many approved tweets exist,
    and generates NEW tweets with different angles.
    """
    attempt = state.get("attempt", 0) + 1
    approved = state.get("approved_tweets", [])

    if attempt == 1:
        result = writer_agent.invoke(
            {"messages": [HumanMessage(content="Analyze today's market and write tweets.")]}
        )
    else:
        batch_msg = WRITER_BATCH_TEMPLATE.format(
            approved_count=len(approved),
            remaining=TOTAL_NEEDED - len(approved),
            total_needed=TOTAL_NEEDED,
            feedback=state.get("audit_feedback", ""),
            batch_size=TWEET_BATCH_SIZE,
        )
        result = writer_agent.invoke(
            {"messages": [HumanMessage(content=batch_msg)]}
        )

    last_message = result["messages"][-1]
    content = last_message.content if isinstance(last_message, AIMessage) else str(last_message)
    tweets = _parse_tweets(content)

    market_data = state.get("market_data", {})
    news_articles = state.get("news_articles", [])

    if attempt == 1:
        for msg in result["messages"]:
            if hasattr(msg, "content") and isinstance(msg.content, str):
                try:
                    parsed = json.loads(msg.content)
                    if isinstance(parsed, dict) and "indices" in parsed:
                        market_data = parsed
                    elif isinstance(parsed, list) and len(parsed) > 0 and "title" in parsed[0]:
                        news_articles = parsed
                except (json.JSONDecodeError, TypeError, IndexError):
                    pass

    return {
        "draft_tweets": tweets,
        "market_data": market_data,
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
    news_articles = state.get("news_articles", [])
    approved = list(state.get("approved_tweets", []))

    audit_input = (
        f"TWEET DRAFTS:\n{json.dumps(drafts, indent=2)}\n\n"
        f"MARKET DATA:\n{json.dumps(market_data, indent=2)}\n\n"
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
        f"Score {t['score']}/10: \"{t['tweet'][:80]}...\" -- {t['feedback']}"
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
    1. One premium thread (1000-1500 chars) with narrative arc
    2. One standalone punchy tweet (under 280 chars)
    """
    approved = state.get("approved_tweets", [])
    market_data = state.get("market_data", {})
    news_articles = state.get("news_articles", [])

    top_tweets = sorted(approved, key=lambda t: t.get("score", 0), reverse=True)
    top_tweets = top_tweets[:TOTAL_NEEDED]

    synth_input = (
        f"APPROVED TWEETS (your raw material -- combine the best elements):\n"
        + "\n\n".join(
            f"[Score {t.get('score', '?')}/12] {t.get('tweet', '')}"
            for t in top_tweets
        )
        + f"\n\nMARKET DATA:\n{json.dumps(market_data, indent=2)}"
        + f"\n\nNEWS ARTICLES:\n{json.dumps(news_articles[:10], indent=2)}"
    )

    response = synthesizer_llm.invoke([
        SystemMessage(content=SYNTHESIZER_PROMPT),
        HumanMessage(content=synth_input),
    ])

    final = _parse_synthesis(response.content)

    return {
        "final_tweets": final,
        "approved_tweets": top_tweets,
    }


def _parse_synthesis(content: str) -> list[dict]:
    """Parse synthesizer JSON response into final tweets list."""
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        data = json.loads(content[start:end])
        result = []
        if data.get("premium_thread"):
            result.append({
                "tweet": data["premium_thread"],
                "type": "premium_thread",
                "score": 12,
            })
        if data.get("standalone_tweet"):
            result.append({
                "tweet": data["standalone_tweet"],
                "type": "standalone",
                "score": 12,
            })
        if result:
            return result
    except (ValueError, json.JSONDecodeError):
        pass

    return [{"tweet": content.strip(), "type": "raw", "score": 0}]


# ---------------------------------------------------------------------------
# Deliver node
# ---------------------------------------------------------------------------

def deliver_node(state: AgentState) -> dict:
    """Send synthesized tweets + individual approved tweets via email."""
    final_tweets = state.get("final_tweets", [])
    approved = state.get("approved_tweets", [])
    market_data = state.get("market_data", {})
    news_articles = state.get("news_articles", [])

    all_tweets = final_tweets + approved

    result = send_email.invoke({
        "scored_tweets": all_tweets,
        "market_data": market_data,
        "news_articles": news_articles,
    })

    return {"audit_feedback": f"Email result: {result}"}


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

    graph.add_node("writer", writer_node)
    graph.add_node("auditor", auditor_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("deliver", deliver_node)

    graph.set_entry_point("writer")

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
