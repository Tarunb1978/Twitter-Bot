"""
Email tool -- sends tweet drafts to Outlook via SMTP.

The email contains:
- All 5 scored tweet drafts (sorted lowest score first, best last)
- Market data summary used by the agent
- Source news links for fact-checking before posting
"""

import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from langchain_core.tools import tool

from config import (
    EMAIL_ADDRESS,
    EMAIL_PASSWORD,
    EMAIL_SMTP_PORT,
    EMAIL_SMTP_SERVER,
    EMAIL_TO,
)


def _build_email_body(
    scored_tweets: list[dict],
    market_data: dict | None = None,
    news_articles: list[dict] | None = None,
) -> str:
    """
    Build a clean HTML email body with scored tweets and sources.

    scored_tweets: list of {"tweet": str, "score": int, "feedback": str}
                   already sorted by score ascending (lowest first).
    """
    lines = []

    synthesized = [t for t in scored_tweets if t.get("type") in ("premium_thread", "standalone")]
    individual = [t for t in scored_tweets if t.get("type") not in ("premium_thread", "standalone")]

    if synthesized:
        lines.append("<h2>READY TO POST</h2>")
        lines.append("<hr>")
        for item in synthesized:
            tweet_type = item.get("type", "")
            tweet = item.get("tweet", "")
            char_count = len(tweet)
            label = "PREMIUM THREAD" if tweet_type == "premium_thread" else "STANDALONE TWEET"
            lines.append(f"<h3>{label} ({char_count} chars)</h3>")
            lines.append(
                '<div style="background:#eafaf1;padding:16px;border-left:6px solid'
                ' #27ae60;margin:8px 0;font-size:17px;line-height:1.6;">'
            )
            lines.append(tweet.replace("\n", "<br>"))
            lines.append("</div>")

    if individual:
        lines.append("<hr>")
        lines.append("<h2>Individual Approved Tweets (raw material)</h2>")
        lines.append("<p><em>These were used to create the premium output above</em></p>")
        for i, item in enumerate(individual, 1):
            score = item.get("score", "?")
            tweet = item.get("tweet", "")
            feedback = item.get("feedback", "")
            char_count = len(tweet)
            lines.append(f"<h3>#{i} &mdash; Score: {score}/12</h3>")
            lines.append(
                f'<div style="background:#f4f4f4;padding:12px;border-left:4px solid'
                f' {"#27ae60" if score >= 9 else "#e67e22" if score >= 7 else "#e74c3c"};'
                f'margin:8px 0;font-size:15px;">'
            )
            lines.append(f"{tweet}")
            lines.append("</div>")
            lines.append(f"<p><small>{char_count} chars | {feedback}</small></p>")

    if market_data:
        lines.append("<hr>")
        lines.append("<h3>Market Data Used</h3>")
        lines.append("<pre style='font-size:13px;'>")

        indices = market_data.get("indices", {})
        if indices:
            lines.append("INDICES:")
            for name, data in indices.items():
                lines.append(f"  {name}: {data['price']} ({data['change_pct']:+.2f}%)")

        movers = market_data.get("top_movers", {})
        if movers.get("gainers"):
            lines.append("\nTOP GAINERS:")
            for s in movers["gainers"]:
                lines.append(f"  {s['ticker']}: {s['price']} ({s['change_pct']:+.2f}%)")
        if movers.get("losers"):
            lines.append("\nTOP LOSERS:")
            for s in movers["losers"]:
                lines.append(f"  {s['ticker']}: {s['price']} ({s['change_pct']:+.2f}%)")

        global_data = market_data.get("global", {})
        if global_data:
            lines.append("\nGLOBAL:")
            for name, data in global_data.items():
                if name.startswith("_") or not isinstance(data, dict):
                    continue
                lines.append(f"  {name}: {data['price']} ({data['change_pct']:+.2f}%)")

        lines.append("</pre>")

    if news_articles:
        lines.append("<hr>")
        lines.append("<h3>News Sources (for fact-checking)</h3>")
        lines.append("<ul>")
        for article in news_articles:
            title = article.get("title", "Untitled")
            url = article.get("url", "")
            source = article.get("source", "")
            provider = article.get("provider", "")
            sentiment = article.get("sentiment")
            sent_str = f" | sentiment: {sentiment}" if sentiment is not None else ""
            lines.append(
                f'<li><a href="{url}">{title}</a>'
                f" <small>({source} via {provider}{sent_str})</small></li>"
            )
        lines.append("</ul>")

    return "\n".join(lines)


@tool
def send_email(
    scored_tweets: list[dict],
    market_data: dict | None = None,
    news_articles: list[dict] | None = None,
) -> str:
    """Send scored tweet drafts via email. Returns 'sent' on success or an error message."""
    if not all([EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_TO]):
        return "Email not configured -- check .env"

    today = datetime.now().strftime("%d %b %Y")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Daily Tweet Drafts -- {today}"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_TO

    body_html = _build_email_body(scored_tweets, market_data, news_articles)
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, EMAIL_TO, msg.as_string())
        return "sent"
    except Exception as e:
        return f"Email failed: {e}"
