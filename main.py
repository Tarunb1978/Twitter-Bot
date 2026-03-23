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
import sys
from datetime import datetime

from agent import build_graph

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


def main():
    logger.info("Starting Tweet Agent...")

    graph = build_graph()

    initial_state = {
        "market_data": {},
        "news_articles": [],
        "draft_tweets": [],
        "scored_tweets": [],
        "approved_tweets": [],
        "final_tweets": [],
        "audit_feedback": "",
        "attempt": 0,
    }

    try:
        result = graph.invoke(initial_state)

        attempt = result.get("attempt", 1)
        final = result.get("final_tweets", [])
        approved_count = len(result.get("approved_tweets", []))

        logger.info(f"Completed in {attempt} batch(es), {approved_count} tweets approved")

        for t in final:
            tweet_type = t.get("type", "unknown")
            tweet_text = t.get("tweet", "N/A")[:200]
            logger.info(f"[{tweet_type}] {tweet_text}...")

        logger.info(f"Email status: {result.get('audit_feedback', 'unknown')}")

    except Exception:
        logger.exception("Agent failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
