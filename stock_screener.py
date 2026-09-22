"""
Stock Screening Agent
======================
Starts from today's largest-%-declining stocks (the primary filter), then
narrows that list down to ones where:
  1. Majority of analysts rate the stock Buy / Overweight / Outperform
  2. Current price is at or near its 52-week low

Run manually:
    python3 stock_screener.py

Intended schedule: 9:45am and 10:00am ET, weekdays.
See "SCHEDULING" section at the bottom of this file for cron/Task Scheduler setup.

Requires:
    pip install yfinance pandas anthropic

For the LLM reasoning step, set an ANTHROPIC_API_KEY environment variable.
Without it, the script still runs the rule-based screen and just skips reasoning.
"""

import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

# How close to the 52-week low counts as "near" (5% = within 5% above the low)
NEAR_LOW_THRESHOLD = 0.05

# Minimum share of analysts that must be bullish (Buy/Overweight/Outperform
# are all bucketed into yfinance's "strongBuy" + "buy" categories)
MIN_BULLISH_PCT = 0.50

# Minimum number of analysts covering the stock, to avoid noise from thin coverage
MIN_ANALYST_COUNT = 3

OUTPUT_DIR = "."  # change to a folder you want CSVs saved into

# --- LLM reasoning ---
# Adds a short AI-generated thesis/red-flags note for each stock that passes
# the rule-based screen. Requires: pip install anthropic
# and an ANTHROPIC_API_KEY environment variable set.
# Controlled by the RUN_REASONING env var so it can be turned off per-deployment
# (e.g. run reasoning only on the 9:45am job, skip it on the 10am confirmation run,
# to cut API cost roughly in half).
ENABLE_LLM_REASONING = os.environ.get("RUN_REASONING", "true").strip().lower() != "false"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
MAX_STOCKS_TO_REASON_ABOUT = 15  # cap API calls/cost if a lot of stocks pass

# --- Eastern-time self-check ---
# The host that triggers this script (e.g. Render's cron) schedules in UTC and
# won't auto-adjust for US Daylight Saving Time. Rather than rely on the
# trigger time being exactly right, the script checks the real Eastern time
# itself and only proceeds if it's actually one of the target run times
# (within a tolerance window). This makes the schedule self-correcting
# through DST changes, at the cost of the trigger needing to fire at least
# once somewhere inside each tolerance window.
TARGET_TIMES_ET = [(9, 45), (10, 0)]  # (hour, minute) in America/New_York
TOLERANCE_MINUTES = 10


def is_target_time_et():
    """True if the current Eastern time falls within TOLERANCE_MINUTES of
    any entry in TARGET_TIMES_ET."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    now_minutes = now_et.hour * 60 + now_et.minute
    for hour, minute in TARGET_TIMES_ET:
        target_minutes = hour * 60 + minute
        if abs(now_minutes - target_minutes) <= TOLERANCE_MINUTES:
            return True, now_et
    return False, now_et


# ----------------------------------------------------------------------
# UNIVERSE
# ----------------------------------------------------------------------
#
# The PRIMARY filter: today's largest-%-declining stocks. Everything else
# (analyst ratings, 52-week-low proximity) is checked only within this set —
# a stock that isn't among today's big decliners never gets considered,
# no matter how bullish its ratings are.

DECLINERS_COUNT = 250  # how many of today's biggest decliners to pull


def get_day_losers(count=DECLINERS_COUNT):
    """Pulls today's largest % decliners via Yahoo Finance's predefined
    'day_losers' screener (through yfinance). This is the starting universe —
    everything downstream filters further within this list."""
    try:
        result = yf.screen("day_losers", count=count)
        quotes = result.get("quotes", [])
        tickers = [q["symbol"] for q in quotes if q.get("symbol")]
        if not tickers:
            print("day_losers screen returned no tickers — falling back to S&P 500.")
            return get_sp500_tickers()
        return tickers
    except Exception as e:
        print(f"day_losers screen failed ({e}) — falling back to S&P 500.")
        return get_sp500_tickers()


def get_sp500_tickers():
    """Fallback universe: current S&P 500 constituents from Wikipedia.
    Used only if the day_losers screen is unavailable."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


# ----------------------------------------------------------------------
# SCREENING LOGIC
# ----------------------------------------------------------------------

def screen_stock(ticker_symbol):
    """Returns a dict of screen results for a ticker, or None if it doesn't pass."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        low_52wk = info.get("fiftyTwoWeekLow")
        if not current_price or not low_52wk:
            return None

        day_change_pct = info.get("regularMarketChangePercent")

        pct_above_low = (current_price - low_52wk) / low_52wk
        if pct_above_low > NEAR_LOW_THRESHOLD:
            return None  # not near the 52-week low

        rec = ticker.recommendations_summary
        if rec is None or rec.empty:
            return None

        latest = rec.iloc[0]  # most recent period (usually labeled "0m")
        strong_buy = int(latest.get("strongBuy", 0))
        buy = int(latest.get("buy", 0))
        hold = int(latest.get("hold", 0))
        sell = int(latest.get("sell", 0))
        strong_sell = int(latest.get("strongSell", 0))

        total = strong_buy + buy + hold + sell + strong_sell
        if total < MIN_ANALYST_COUNT:
            return None

        bullish = strong_buy + buy
        bullish_pct = bullish / total
        if bullish_pct <= MIN_BULLISH_PCT:
            return None

        return {
            "ticker": ticker_symbol,
            "company": info.get("shortName", ""),
            "current_price": round(current_price, 2),
            "day_change_pct": round(day_change_pct, 2) if day_change_pct is not None else None,
            "52wk_low": round(low_52wk, 2),
            "pct_above_low": round(pct_above_low * 100, 2),
            "bullish_pct": round(bullish_pct * 100, 1),
            "strongBuy": strong_buy,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "strongSell": strong_sell,
            "analyst_count": total,
        }
    except Exception:
        return None


def run_screen(tickers, delay=0.1):
    """Screens every ticker, with a small delay to avoid rate-limiting."""
    results = []
    for t in tickers:
        result = screen_stock(t)
        if result:
            results.append(result)
        time.sleep(delay)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("pct_above_low")
    return df


# ----------------------------------------------------------------------
# LLM REASONING
# ----------------------------------------------------------------------

def get_recent_headlines(ticker_symbol, max_headlines=5):
    """Pulls a handful of recent headlines for a ticker via yfinance."""
    try:
        news = yf.Ticker(ticker_symbol).news or []
        headlines = []
        for item in news[:max_headlines]:
            title = item.get("content", {}).get("title") or item.get("title")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def get_llm_reasoning(client, row):
    """
    Asks Claude to reason over one stock's metrics + recent headlines:
    why it might be showing up here, and what to watch out for.
    Returns a dict: {"reasoning": str, "flag": str, "flag_reason": str}
    flag is one of: "structural_concern", "macro_or_sector", "unclear"
    Returns None on total failure.
    """
    headlines = get_recent_headlines(row["ticker"])
    headline_block = "\n".join(f"- {h}" for h in headlines) if headlines else "(no recent headlines found)"

    prompt = f"""You are helping a trader sanity-check a stock screen. This stock
passed a rule-based screen: majority-bullish analyst ratings, but the price is
near its 52-week low (a "falling knife vs. buying opportunity" setup).

Ticker: {row['ticker']} ({row.get('company', '')})
Current price: ${row['current_price']}
52-week low: ${row['52wk_low']} ({row['pct_above_low']}% above it)
Analyst ratings: {row['strongBuy']} strong buy, {row['buy']} buy, {row['hold']} hold, \
{row['sell']} sell, {row['strongSell']} strong sell ({row['bullish_pct']}% bullish, \
{row['analyst_count']} analysts)

Recent headlines:
{headline_block}

Classify why this stock is likely near its low, and respond with ONLY a JSON
object (no markdown fences, no preamble) with these exact keys:

"reasoning": 2-3 concise, concrete sentences on why it's near its low despite
  bullish coverage, and what's worth checking before treating this as a buy signal.
"flag": one of "structural_concern", "macro_or_sector", or "unclear".
  Use "structural_concern" ONLY if headlines point to something company-specific
  and fundamental — e.g. accounting/restatement issues, guidance cuts, missed
  earnings, executive departures, regulatory/legal trouble, loss of a major
  customer, debt/liquidity problems. Use "macro_or_sector" if the decline looks
  driven by broad market, sector, or rate-driven selling with no company-specific
  red flag. Use "unclear" if headlines are absent or give no real signal either way.
"flag_reason": one short phrase (under 12 words) naming the specific headline
  detail that drove the flag choice, or "no clear signal in headlines" if unclear.

Be skeptical and concrete, not generic."""

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return {
            "reasoning": parsed.get("reasoning", "").strip(),
            "flag": parsed.get("flag", "unclear").strip(),
            "flag_reason": parsed.get("flag_reason", "").strip(),
        }
    except Exception as e:
        return {"reasoning": f"(reasoning failed: {e})", "flag": "unclear", "flag_reason": ""}


def add_llm_reasoning(df):
    """Adds an 'llm_reasoning' column to the results DataFrame, in place of a copy."""
    if df.empty:
        return df

    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed (pip install anthropic) — skipping LLM reasoning.")
        return df

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — skipping LLM reasoning.")
        return df

    client = anthropic.Anthropic(api_key=api_key)

    df = df.copy()
    df["llm_reasoning"] = ""
    df["risk_flag"] = ""
    df["flag_reason"] = ""
    subset = df.head(MAX_STOCKS_TO_REASON_ABOUT)
    for idx, row in subset.iterrows():
        print(f"  Reasoning about {row['ticker']}...")
        result = get_llm_reasoning(client, row)
        df.at[idx, "llm_reasoning"] = result["reasoning"]
        df.at[idx, "risk_flag"] = result["flag"]
        df.at[idx, "flag_reason"] = result["flag_reason"]
        time.sleep(0.5)  # be polite to the API

    return df


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    on_time, now_et = is_target_time_et()
    if not on_time:
        print(f"[{now_et}] Not a scheduled run time (ET) — skipping. "
              f"Target times: {TARGET_TIMES_ET} ET, ±{TOLERANCE_MINUTES} min.")
        sys.exit(0)

    print(f"[{now_et}] On schedule — starting screen...")
    tickers = get_day_losers()
    print(f"Screening {len(tickers)} of today's largest decliners...")
    df = run_screen(tickers)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{OUTPUT_DIR}/stock_screen_{timestamp}.csv"

    if df.empty:
        print("No stocks matched the criteria today.")
    else:
        print(f"Found {len(df)} matches.")
        if ENABLE_LLM_REASONING:
            df = add_llm_reasoning(df)
            if "risk_flag" in df.columns:
                # Surface structural concerns first, then macro/sector, then unclear
                flag_order = {"structural_concern": 0, "macro_or_sector": 1, "unclear": 2, "": 3}
                df["_sort"] = df["risk_flag"].map(lambda f: flag_order.get(f, 3))
                df = df.sort_values(["_sort", "pct_above_low"]).drop(columns="_sort")

        df.to_csv(filename, index=False)
        print(f"Saved to {filename}\n")

        # Print a readable summary (full table goes to the CSV)
        FLAG_LABEL = {
            "structural_concern": "⚠️  STRUCTURAL CONCERN",
            "macro_or_sector": "macro/sector-driven",
            "unclear": "unclear signal",
        }
        for _, row in df.iterrows():
            print(f"\n{row['ticker']} ({row.get('company', '')}) — "
                  f"${row['current_price']} ({row.get('day_change_pct')}% today), "
                  f"{row['pct_above_low']}% above 52wk low, "
                  f"{row['bullish_pct']}% bullish ({row['analyst_count']} analysts)")
            if row.get("risk_flag"):
                label = FLAG_LABEL.get(row["risk_flag"], row["risk_flag"])
                reason = f" — {row['flag_reason']}" if row.get("flag_reason") else ""
                print(f"  [{label}]{reason}")
            if row.get("llm_reasoning"):
                print(f"  → {row['llm_reasoning']}")


# ----------------------------------------------------------------------
# SCHEDULING (macOS / Linux — cron)
# ----------------------------------------------------------------------
# 1. Find your python3 path:      which python3
# 2. Edit your crontab:           crontab -e
# 3. Add these two lines (adjust paths; cron uses your SYSTEM's local time,
#    so if your machine isn't set to US/Eastern, convert 9:45/10:00 ET first):
#
#    45 9 * * 1-5  cd /path/to/script && /usr/bin/python3 stock_screener.py >> screen.log 2>&1
#    0 10 * * 1-5  cd /path/to/script && /usr/bin/python3 stock_screener.py >> screen.log 2>&1
#
#    (1-5 = Monday-Friday only)
#
# ----------------------------------------------------------------------
# SCHEDULING (Windows — Task Scheduler)
# ----------------------------------------------------------------------
# Create two Basic Tasks that run daily, weekdays only, at 9:45 AM and
# 10:00 AM (in your local time zone — convert from ET if needed), each
# with action: "python.exe" and argument: "C:\path\to\stock_screener.py"
#
# ----------------------------------------------------------------------
# SCHEDULING (cloud — no computer needed)
# ----------------------------------------------------------------------
# GitHub Actions example (.github/workflows/screen.yml), cron is UTC so
# 9:45/10:00 ET becomes 13:45/14:00 UTC during EDT (summer) or 14:45/15:00
# UTC during EST (winter):
#
#   on:
#     schedule:
#       - cron: '45 13 * * 1-5'
#       - cron: '0 14 * * 1-5'
#   jobs:
#     screen:
#       runs-on: ubuntu-latest
#       steps:
#         - uses: actions/checkout@v4
#         - uses: actions/setup-python@v5
#           with: { python-version: '3.11' }
#         - run: pip install yfinance pandas
#         - run: python stock_screener.py
#         - uses: actions/upload-artifact@v4
#           with: { name: screen-results, path: stock_screen_*.csv }





"""
Stock Screening Agent
======================
Screens a universe of stocks (default: S&P 500) for:
  1. Majority of analysts rate the stock Buy / Overweight / Outperform
  2. Current price is at or near its 52-week low

Run manually:
    python3 stock_screener.py

Intended schedule: 9:45am and 10:00am ET, weekdays.
See "SCHEDULING" section at the bottom of this file for cron/Task Scheduler setup.

Requires:
    pip install yfinance pandas anthropic

For the LLM reasoning step, set an ANTHROPIC_API_KEY environment variable.
Without it, the script still runs the rule-based screen and just skips reasoning.
"""

import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

# How close to the 52-week low counts as "near" (5% = within 5% above the low)
NEAR_LOW_THRESHOLD = 0.05

# Minimum share of analysts that must be bullish (Buy/Overweight/Outperform
# are all bucketed into yfinance's "strongBuy" + "buy" categories)
MIN_BULLISH_PCT = 0.50

# Minimum number of analysts covering the stock, to avoid noise from thin coverage
MIN_ANALYST_COUNT = 3

OUTPUT_DIR = "."  # change to a folder you want CSVs saved into

# --- LLM reasoning ---
# Adds a short AI-generated thesis/red-flags note for each stock that passes
# the rule-based screen. Requires: pip install anthropic
# and an ANTHROPIC_API_KEY environment variable set.
# Controlled by the RUN_REASONING env var so it can be turned off per-deployment
# (e.g. run reasoning only on the 9:45am job, skip it on the 10am confirmation run,
# to cut API cost roughly in half).
ENABLE_LLM_REASONING = os.environ.get("RUN_REASONING", "true").strip().lower() != "false"
ANTHROPIC_MODEL = "claude-sonnet-4-6"
MAX_STOCKS_TO_REASON_ABOUT = 15  # cap API calls/cost if a lot of stocks pass

# --- Eastern-time self-check ---
# The host that triggers this script (e.g. Render's cron) schedules in UTC and
# won't auto-adjust for US Daylight Saving Time. Rather than rely on the
# trigger time being exactly right, the script checks the real Eastern time
# itself and only proceeds if it's actually one of the target run times
# (within a tolerance window). This makes the schedule self-correcting
# through DST changes, at the cost of the trigger needing to fire at least
# once somewhere inside each tolerance window.
TARGET_TIMES_ET = [(9, 45), (10, 0)]  # (hour, minute) in America/New_York
TOLERANCE_MINUTES = 10


def is_target_time_et():
    """True if the current Eastern time falls within TOLERANCE_MINUTES of
    any entry in TARGET_TIMES_ET."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    now_minutes = now_et.hour * 60 + now_et.minute
    for hour, minute in TARGET_TIMES_ET:
        target_minutes = hour * 60 + minute
        if abs(now_minutes - target_minutes) <= TOLERANCE_MINUTES:
            return True, now_et
    return False, now_et


# ----------------------------------------------------------------------
# UNIVERSE
# ----------------------------------------------------------------------

def get_sp500_tickers():
    """Pulls the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


# ----------------------------------------------------------------------
# SCREENING LOGIC
# ----------------------------------------------------------------------

def screen_stock(ticker_symbol):
    """Returns a dict of screen results for a ticker, or None if it doesn't pass."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        low_52wk = info.get("fiftyTwoWeekLow")
        if not current_price or not low_52wk:
            return None

        pct_above_low = (current_price - low_52wk) / low_52wk
        if pct_above_low > NEAR_LOW_THRESHOLD:
            return None  # not near the 52-week low

        rec = ticker.recommendations_summary
        if rec is None or rec.empty:
            return None

        latest = rec.iloc[0]  # most recent period (usually labeled "0m")
        strong_buy = int(latest.get("strongBuy", 0))
        buy = int(latest.get("buy", 0))
        hold = int(latest.get("hold", 0))
        sell = int(latest.get("sell", 0))
        strong_sell = int(latest.get("strongSell", 0))

        total = strong_buy + buy + hold + sell + strong_sell
        if total < MIN_ANALYST_COUNT:
            return None

        bullish = strong_buy + buy
        bullish_pct = bullish / total
        if bullish_pct <= MIN_BULLISH_PCT:
            return None

        return {
            "ticker": ticker_symbol,
            "company": info.get("shortName", ""),
            "current_price": round(current_price, 2),
            "52wk_low": round(low_52wk, 2),
            "pct_above_low": round(pct_above_low * 100, 2),
            "bullish_pct": round(bullish_pct * 100, 1),
            "strongBuy": strong_buy,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "strongSell": strong_sell,
            "analyst_count": total,
        }
    except Exception:
        return None


def run_screen(tickers, delay=0.1):
    """Screens every ticker, with a small delay to avoid rate-limiting."""
    results = []
    for t in tickers:
        result = screen_stock(t)
        if result:
            results.append(result)
        time.sleep(delay)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("pct_above_low")
    return df


# ----------------------------------------------------------------------
# LLM REASONING
# ----------------------------------------------------------------------

def get_recent_headlines(ticker_symbol, max_headlines=5):
    """Pulls a handful of recent headlines for a ticker via yfinance."""
    try:
        news = yf.Ticker(ticker_symbol).news or []
        headlines = []
        for item in news[:max_headlines]:
            title = item.get("content", {}).get("title") or item.get("title")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def get_llm_reasoning(client, row):
    """
    Asks Claude to reason over one stock's metrics + recent headlines:
    why it might be showing up here, and what to watch out for.
    Returns a dict: {"reasoning": str, "flag": str, "flag_reason": str}
    flag is one of: "structural_concern", "macro_or_sector", "unclear"
    Returns None on total failure.
    """
    headlines = get_recent_headlines(row["ticker"])
    headline_block = "\n".join(f"- {h}" for h in headlines) if headlines else "(no recent headlines found)"

    prompt = f"""You are helping a trader sanity-check a stock screen. This stock
passed a rule-based screen: majority-bullish analyst ratings, but the price is
near its 52-week low (a "falling knife vs. buying opportunity" setup).

Ticker: {row['ticker']} ({row.get('company', '')})
Current price: ${row['current_price']}
52-week low: ${row['52wk_low']} ({row['pct_above_low']}% above it)
Analyst ratings: {row['strongBuy']} strong buy, {row['buy']} buy, {row['hold']} hold, \
{row['sell']} sell, {row['strongSell']} strong sell ({row['bullish_pct']}% bullish, \
{row['analyst_count']} analysts)

Recent headlines:
{headline_block}

Classify why this stock is likely near its low, and respond with ONLY a JSON
object (no markdown fences, no preamble) with these exact keys:

"reasoning": 2-3 concise, concrete sentences on why it's near its low despite
  bullish coverage, and what's worth checking before treating this as a buy signal.
"flag": one of "structural_concern", "macro_or_sector", or "unclear".
  Use "structural_concern" ONLY if headlines point to something company-specific
  and fundamental — e.g. accounting/restatement issues, guidance cuts, missed
  earnings, executive departures, regulatory/legal trouble, loss of a major
  customer, debt/liquidity problems. Use "macro_or_sector" if the decline looks
  driven by broad market, sector, or rate-driven selling with no company-specific
  red flag. Use "unclear" if headlines are absent or give no real signal either way.
"flag_reason": one short phrase (under 12 words) naming the specific headline
  detail that drove the flag choice, or "no clear signal in headlines" if unclear.

Be skeptical and concrete, not generic."""

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return {
            "reasoning": parsed.get("reasoning", "").strip(),
            "flag": parsed.get("flag", "unclear").strip(),
            "flag_reason": parsed.get("flag_reason", "").strip(),
        }
    except Exception as e:
        return {"reasoning": f"(reasoning failed: {e})", "flag": "unclear", "flag_reason": ""}


def add_llm_reasoning(df):
    """Adds an 'llm_reasoning' column to the results DataFrame, in place of a copy."""
    if df.empty:
        return df

    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed (pip install anthropic) — skipping LLM reasoning.")
        return df

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — skipping LLM reasoning.")
        return df

    client = anthropic.Anthropic(api_key=api_key)

    df = df.copy()
    df["llm_reasoning"] = ""
    df["risk_flag"] = ""
    df["flag_reason"] = ""
    subset = df.head(MAX_STOCKS_TO_REASON_ABOUT)
    for idx, row in subset.iterrows():
        print(f"  Reasoning about {row['ticker']}...")
        result = get_llm_reasoning(client, row)
        df.at[idx, "llm_reasoning"] = result["reasoning"]
        df.at[idx, "risk_flag"] = result["flag"]
        df.at[idx, "flag_reason"] = result["flag_reason"]
        time.sleep(0.5)  # be polite to the API

    return df


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    on_time, now_et = is_target_time_et()
    if not on_time:
        print(f"[{now_et}] Not a scheduled run time (ET) — skipping. "
              f"Target times: {TARGET_TIMES_ET} ET, ±{TOLERANCE_MINUTES} min.")
        sys.exit(0)

    print(f"[{now_et}] On schedule — starting screen...")
    tickers = get_sp500_tickers()
    df = run_screen(tickers)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{OUTPUT_DIR}/stock_screen_{timestamp}.csv"

    if df.empty:
        print("No stocks matched the criteria today.")
    else:
        print(f"Found {len(df)} matches.")
        if ENABLE_LLM_REASONING:
            df = add_llm_reasoning(df)
            if "risk_flag" in df.columns:
                # Surface structural concerns first, then macro/sector, then unclear
                flag_order = {"structural_concern": 0, "macro_or_sector": 1, "unclear": 2, "": 3}
                df["_sort"] = df["risk_flag"].map(lambda f: flag_order.get(f, 3))
                df = df.sort_values(["_sort", "pct_above_low"]).drop(columns="_sort")

        df.to_csv(filename, index=False)
        print(f"Saved to {filename}\n")

        # Print a readable summary (full table goes to the CSV)
        FLAG_LABEL = {
            "structural_concern": "⚠️  STRUCTURAL CONCERN",
            "macro_or_sector": "macro/sector-driven",
            "unclear": "unclear signal",
        }
        for _, row in df.iterrows():
            print(f"\n{row['ticker']} ({row.get('company', '')}) — "
                  f"${row['current_price']}, {row['pct_above_low']}% above 52wk low, "
                  f"{row['bullish_pct']}% bullish ({row['analyst_count']} analysts)")
            if row.get("risk_flag"):
                label = FLAG_LABEL.get(row["risk_flag"], row["risk_flag"])
                reason = f" — {row['flag_reason']}" if row.get("flag_reason") else ""
                print(f"  [{label}]{reason}")
            if row.get("llm_reasoning"):
                print(f"  → {row['llm_reasoning']}")


# ----------------------------------------------------------------------
# SCHEDULING (macOS / Linux — cron)
# ----------------------------------------------------------------------
# 1. Find your python3 path:      which python3
# 2. Edit your crontab:           crontab -e
# 3. Add these two lines (adjust paths; cron uses your SYSTEM's local time,
#    so if your machine isn't set to US/Eastern, convert 9:45/10:00 ET first):
#
#    45 9 * * 1-5  cd /path/to/script && /usr/bin/python3 stock_screener.py >> screen.log 2>&1
#    0 10 * * 1-5  cd /path/to/script && /usr/bin/python3 stock_screener.py >> screen.log 2>&1
#
#    (1-5 = Monday-Friday only)
#
# ----------------------------------------------------------------------
# SCHEDULING (Windows — Task Scheduler)
# ----------------------------------------------------------------------
# Create two Basic Tasks that run daily, weekdays only, at 9:45 AM and
# 10:00 AM (in your local time zone — convert from ET if needed), each
# with action: "python.exe" and argument: "C:\path\to\stock_screener.py"
#
# ----------------------------------------------------------------------
# SCHEDULING (cloud — no computer needed)
# ----------------------------------------------------------------------
# GitHub Actions example (.github/workflows/screen.yml), cron is UTC so
# 9:45/10:00 ET becomes 13:45/14:00 UTC during EDT (summer) or 14:45/15:00
# UTC during EST (winter):
#
#   on:
#     schedule:
#       - cron: '45 13 * * 1-5'
#       - cron: '0 14 * * 1-5'
#   jobs:
#     screen:
#       runs-on: ubuntu-latest
#       steps:
#         - uses: actions/checkout@v4
#         - uses: actions/setup-python@v5
#           with: { python-version: '3.11' }
#         - run: pip install yfinance pandas
#         - run: python stock_screener.py
#         - uses: actions/upload-artifact@v4
#           with: { name: screen-results, path: stock_screen_*.csv }


"""
Stock Screening Agent
======================
Screens a universe of stocks (default: S&P 500) for:
  1. Majority of analysts rate the stock Buy / Overweight / Outperform
  2. Current price is at or near its 52-week low

Run manually:
    python3 stock_screener.py

Intended schedule: 9:45am and 10:00am ET, weekdays.
See "SCHEDULING" section at the bottom of this file for cron/Task Scheduler setup.

Requires:
    pip install yfinance pandas anthropic

For the LLM reasoning step, set an ANTHROPIC_API_KEY environment variable.
Without it, the script still runs the rule-based screen and just skips reasoning.
"""

import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

# How close to the 52-week low counts as "near" (10% = within 10% above the low)
NEAR_LOW_THRESHOLD = 0.10

# Minimum share of analysts that must be bullish (Buy/Overweight/Outperform
# are all bucketed into yfinance's "strongBuy" + "buy" categories)
MIN_BULLISH_PCT = 0.50

# Minimum number of analysts covering the stock, to avoid noise from thin coverage
MIN_ANALYST_COUNT = 3

OUTPUT_DIR = "."  # change to a folder you want CSVs saved into

# --- LLM reasoning ---
# Adds a short AI-generated thesis/red-flags note for each stock that passes
# the rule-based screen. Requires: pip install anthropic
# and an ANTHROPIC_API_KEY environment variable set.
ENABLE_LLM_REASONING = True
ANTHROPIC_MODEL = "claude-sonnet-4-6"
MAX_STOCKS_TO_REASON_ABOUT = 15  # cap API calls/cost if a lot of stocks pass

# --- Eastern-time self-check ---
# The host that triggers this script (e.g. Render's cron) schedules in UTC and
# won't auto-adjust for US Daylight Saving Time. Rather than rely on the
# trigger time being exactly right, the script checks the real Eastern time
# itself and only proceeds if it's actually one of the target run times
# (within a tolerance window). This makes the schedule self-correcting
# through DST changes, at the cost of the trigger needing to fire at least
# once somewhere inside each tolerance window.
TARGET_TIMES_ET = [(9, 45), (10, 0)]  # (hour, minute) in America/New_York
TOLERANCE_MINUTES = 10


def is_target_time_et():
    """True if the current Eastern time falls within TOLERANCE_MINUTES of
    any entry in TARGET_TIMES_ET."""
    now_et = datetime.now(ZoneInfo("America/New_York"))
    now_minutes = now_et.hour * 60 + now_et.minute
    for hour, minute in TARGET_TIMES_ET:
        target_minutes = hour * 60 + minute
        if abs(now_minutes - target_minutes) <= TOLERANCE_MINUTES:
            return True, now_et
    return False, now_et


# ----------------------------------------------------------------------
# UNIVERSE
# ----------------------------------------------------------------------

def get_sp500_tickers():
    """Pulls the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


# ----------------------------------------------------------------------
# SCREENING LOGIC
# ----------------------------------------------------------------------

def screen_stock(ticker_symbol):
    """Returns a dict of screen results for a ticker, or None if it doesn't pass."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        low_52wk = info.get("fiftyTwoWeekLow")
        if not current_price or not low_52wk:
            return None

        pct_above_low = (current_price - low_52wk) / low_52wk
        if pct_above_low > NEAR_LOW_THRESHOLD:
            return None  # not near the 52-week low

        rec = ticker.recommendations_summary
        if rec is None or rec.empty:
            return None

        latest = rec.iloc[0]  # most recent period (usually labeled "0m")
        strong_buy = int(latest.get("strongBuy", 0))
        buy = int(latest.get("buy", 0))
        hold = int(latest.get("hold", 0))
        sell = int(latest.get("sell", 0))
        strong_sell = int(latest.get("strongSell", 0))

        total = strong_buy + buy + hold + sell + strong_sell
        if total < MIN_ANALYST_COUNT:
            return None

        bullish = strong_buy + buy
        bullish_pct = bullish / total
        if bullish_pct <= MIN_BULLISH_PCT:
            return None

        return {
            "ticker": ticker_symbol,
            "company": info.get("shortName", ""),
            "current_price": round(current_price, 2),
            "52wk_low": round(low_52wk, 2),
            "pct_above_low": round(pct_above_low * 100, 2),
            "bullish_pct": round(bullish_pct * 100, 1),
            "strongBuy": strong_buy,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "strongSell": strong_sell,
            "analyst_count": total,
        }
    except Exception:
        return None


def run_screen(tickers, delay=0.1):
    """Screens every ticker, with a small delay to avoid rate-limiting."""
    results = []
    for t in tickers:
        result = screen_stock(t)
        if result:
            results.append(result)
        time.sleep(delay)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("pct_above_low")
    return df


# ----------------------------------------------------------------------
# LLM REASONING
# ----------------------------------------------------------------------

def get_recent_headlines(ticker_symbol, max_headlines=5):
    """Pulls a handful of recent headlines for a ticker via yfinance."""
    try:
        news = yf.Ticker(ticker_symbol).news or []
        headlines = []
        for item in news[:max_headlines]:
            title = item.get("content", {}).get("title") or item.get("title")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def get_llm_reasoning(client, row):
    """
    Asks Claude to reason over one stock's metrics + recent headlines:
    why it might be showing up here, and what to watch out for.
    Returns a dict: {"reasoning": str, "flag": str, "flag_reason": str}
    flag is one of: "structural_concern", "macro_or_sector", "unclear"
    Returns None on total failure.
    """
    headlines = get_recent_headlines(row["ticker"])
    headline_block = "\n".join(f"- {h}" for h in headlines) if headlines else "(no recent headlines found)"

    prompt = f"""You are helping a trader sanity-check a stock screen. This stock
passed a rule-based screen: majority-bullish analyst ratings, but the price is
near its 52-week low (a "falling knife vs. buying opportunity" setup).

Ticker: {row['ticker']} ({row.get('company', '')})
Current price: ${row['current_price']}
52-week low: ${row['52wk_low']} ({row['pct_above_low']}% above it)
Analyst ratings: {row['strongBuy']} strong buy, {row['buy']} buy, {row['hold']} hold, \
{row['sell']} sell, {row['strongSell']} strong sell ({row['bullish_pct']}% bullish, \
{row['analyst_count']} analysts)

Recent headlines:
{headline_block}

Classify why this stock is likely near its low, and respond with ONLY a JSON
object (no markdown fences, no preamble) with these exact keys:

"reasoning": 2-3 concise, concrete sentences on why it's near its low despite
  bullish coverage, and what's worth checking before treating this as a buy signal.
"flag": one of "structural_concern", "macro_or_sector", or "unclear".
  Use "structural_concern" ONLY if headlines point to something company-specific
  and fundamental — e.g. accounting/restatement issues, guidance cuts, missed
  earnings, executive departures, regulatory/legal trouble, loss of a major
  customer, debt/liquidity problems. Use "macro_or_sector" if the decline looks
  driven by broad market, sector, or rate-driven selling with no company-specific
  red flag. Use "unclear" if headlines are absent or give no real signal either way.
"flag_reason": one short phrase (under 12 words) naming the specific headline
  detail that drove the flag choice, or "no clear signal in headlines" if unclear.

Be skeptical and concrete, not generic."""

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return {
            "reasoning": parsed.get("reasoning", "").strip(),
            "flag": parsed.get("flag", "unclear").strip(),
            "flag_reason": parsed.get("flag_reason", "").strip(),
        }
    except Exception as e:
        return {"reasoning": f"(reasoning failed: {e})", "flag": "unclear", "flag_reason": ""}


def add_llm_reasoning(df):
    """Adds an 'llm_reasoning' column to the results DataFrame, in place of a copy."""
    if df.empty:
        return df

    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed (pip install anthropic) — skipping LLM reasoning.")
        return df

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — skipping LLM reasoning.")
        return df

    client = anthropic.Anthropic(api_key=api_key)

    df = df.copy()
    df["llm_reasoning"] = ""
    df["risk_flag"] = ""
    df["flag_reason"] = ""
    subset = df.head(MAX_STOCKS_TO_REASON_ABOUT)
    for idx, row in subset.iterrows():
        print(f"  Reasoning about {row['ticker']}...")
        result = get_llm_reasoning(client, row)
        df.at[idx, "llm_reasoning"] = result["reasoning"]
        df.at[idx, "risk_flag"] = result["flag"]
        df.at[idx, "flag_reason"] = result["flag_reason"]
        time.sleep(0.5)  # be polite to the API

    return df


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    on_time, now_et = is_target_time_et()
    if not on_time:
        print(f"[{now_et}] Not a scheduled run time (ET) — skipping. "
              f"Target times: {TARGET_TIMES_ET} ET, ±{TOLERANCE_MINUTES} min.")
        sys.exit(0)

    print(f"[{now_et}] On schedule — starting screen...")
    tickers = get_sp500_tickers()
    df = run_screen(tickers)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{OUTPUT_DIR}/stock_screen_{timestamp}.csv"

    if df.empty:
        print("No stocks matched the criteria today.")
    else:
        print(f"Found {len(df)} matches.")
        if ENABLE_LLM_REASONING:
            df = add_llm_reasoning(df)
            if "risk_flag" in df.columns:
                # Surface structural concerns first, then macro/sector, then unclear
                flag_order = {"structural_concern": 0, "macro_or_sector": 1, "unclear": 2, "": 3}
                df["_sort"] = df["risk_flag"].map(lambda f: flag_order.get(f, 3))
                df = df.sort_values(["_sort", "pct_above_low"]).drop(columns="_sort")

        df.to_csv(filename, index=False)
        print(f"Saved to {filename}\n")

        # Print a readable summary (full table goes to the CSV)
        FLAG_LABEL = {
            "structural_concern": "⚠️  STRUCTURAL CONCERN",
            "macro_or_sector": "macro/sector-driven",
            "unclear": "unclear signal",
        }
        for _, row in df.iterrows():
            print(f"\n{row['ticker']} ({row.get('company', '')}) — "
                  f"${row['current_price']}, {row['pct_above_low']}% above 52wk low, "
                  f"{row['bullish_pct']}% bullish ({row['analyst_count']} analysts)")
            if row.get("risk_flag"):
                label = FLAG_LABEL.get(row["risk_flag"], row["risk_flag"])
                reason = f" — {row['flag_reason']}" if row.get("flag_reason") else ""
                print(f"  [{label}]{reason}")
            if row.get("llm_reasoning"):
                print(f"  → {row['llm_reasoning']}")


# ----------------------------------------------------------------------
# SCHEDULING (macOS / Linux — cron)
# ----------------------------------------------------------------------
# 1. Find your python3 path:      which python3
# 2. Edit your crontab:           crontab -e
# 3. Add these two lines (adjust paths; cron uses your SYSTEM's local time,
#    so if your machine isn't set to US/Eastern, convert 9:45/10:00 ET first):
#
#    45 9 * * 1-5  cd /path/to/script && /usr/bin/python3 stock_screener.py >> screen.log 2>&1
#    0 10 * * 1-5  cd /path/to/script && /usr/bin/python3 stock_screener.py >> screen.log 2>&1
#
#    (1-5 = Monday-Friday only)
#
# ----------------------------------------------------------------------
# SCHEDULING (Windows — Task Scheduler)
# ----------------------------------------------------------------------
# Create two Basic Tasks that run daily, weekdays only, at 9:45 AM and
# 10:00 AM (in your local time zone — convert from ET if needed), each
# with action: "python.exe" and argument: "C:\path\to\stock_screener.py"
#
# ----------------------------------------------------------------------
# SCHEDULING (cloud — no computer needed)
# ----------------------------------------------------------------------
# GitHub Actions example (.github/workflows/screen.yml), cron is UTC so
# 9:45/10:00 ET becomes 13:45/14:00 UTC during EDT (summer) or 14:45/15:00
# UTC during EST (winter):
#
#   on:
#     schedule:
#       - cron: '45 13 * * 1-5'
#       - cron: '0 14 * * 1-5'
#   jobs:
#     screen:
#       runs-on: ubuntu-latest
#       steps:
#         - uses: actions/checkout@v4
#         - uses: actions/setup-python@v5
#           with: { python-version: '3.11' }
#         - run: pip install yfinance pandas
#         - run: python stock_screener.py
#         - uses: actions/upload-artifact@v4
#           with: { name: screen-results, path: stock_screen_*.csv }

"""
Stock Screening Agent
======================
Screens a universe of stocks (default: S&P 500) for:
  1. Majority of analysts rate the stock Buy / Overweight / Outperform
  2. Current price is at or near its 52-week low

Run manually:
    python3 stock_screener.py

Intended schedule: 9:45am and 10:00am ET, weekdays.
See "SCHEDULING" section at the bottom of this file for cron/Task Scheduler setup.

Requires:
    pip install yfinance pandas anthropic

For the LLM reasoning step, set an ANTHROPIC_API_KEY environment variable.
Without it, the script still runs the rule-based screen and just skips reasoning.
"""

import json
import os
import time
from datetime import datetime

import pandas as pd
import yfinance as yf

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

# How close to the 52-week low counts as "near" (10% = within 10% above the low)
NEAR_LOW_THRESHOLD = 0.10

# Minimum share of analysts that must be bullish (Buy/Overweight/Outperform
# are all bucketed into yfinance's "strongBuy" + "buy" categories)
MIN_BULLISH_PCT = 0.50

# Minimum number of analysts covering the stock, to avoid noise from thin coverage
MIN_ANALYST_COUNT = 3

OUTPUT_DIR = "."  # change to a folder you want CSVs saved into

# --- LLM reasoning ---
# Adds a short AI-generated thesis/red-flags note for each stock that passes
# the rule-based screen. Requires: pip install anthropic
# and an ANTHROPIC_API_KEY environment variable set.
ENABLE_LLM_REASONING = True
ANTHROPIC_MODEL = "claude-sonnet-4-6"
MAX_STOCKS_TO_REASON_ABOUT = 15  # cap API calls/cost if a lot of stocks pass


# ----------------------------------------------------------------------
# UNIVERSE
# ----------------------------------------------------------------------

def get_sp500_tickers():
    """Pulls the current S&P 500 constituent list from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    return table["Symbol"].str.replace(".", "-", regex=False).tolist()


# ----------------------------------------------------------------------
# SCREENING LOGIC
# ----------------------------------------------------------------------

def screen_stock(ticker_symbol):
    """Returns a dict of screen results for a ticker, or None if it doesn't pass."""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        low_52wk = info.get("fiftyTwoWeekLow")
        if not current_price or not low_52wk:
            return None

        pct_above_low = (current_price - low_52wk) / low_52wk
        if pct_above_low > NEAR_LOW_THRESHOLD:
            return None  # not near the 52-week low

        rec = ticker.recommendations_summary
        if rec is None or rec.empty:
            return None

        latest = rec.iloc[0]  # most recent period (usually labeled "0m")
        strong_buy = int(latest.get("strongBuy", 0))
        buy = int(latest.get("buy", 0))
        hold = int(latest.get("hold", 0))
        sell = int(latest.get("sell", 0))
        strong_sell = int(latest.get("strongSell", 0))

        total = strong_buy + buy + hold + sell + strong_sell
        if total < MIN_ANALYST_COUNT:
            return None

        bullish = strong_buy + buy
        bullish_pct = bullish / total
        if bullish_pct <= MIN_BULLISH_PCT:
            return None

        return {
            "ticker": ticker_symbol,
            "company": info.get("shortName", ""),
            "current_price": round(current_price, 2),
            "52wk_low": round(low_52wk, 2),
            "pct_above_low": round(pct_above_low * 100, 2),
            "bullish_pct": round(bullish_pct * 100, 1),
            "strongBuy": strong_buy,
            "buy": buy,
            "hold": hold,
            "sell": sell,
            "strongSell": strong_sell,
            "analyst_count": total,
        }
    except Exception:
        return None


def run_screen(tickers, delay=0.1):
    """Screens every ticker, with a small delay to avoid rate-limiting."""
    results = []
    for t in tickers:
        result = screen_stock(t)
        if result:
            results.append(result)
        time.sleep(delay)
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values("pct_above_low")
    return df


# ----------------------------------------------------------------------
# LLM REASONING
# ----------------------------------------------------------------------

def get_recent_headlines(ticker_symbol, max_headlines=5):
    """Pulls a handful of recent headlines for a ticker via yfinance."""
    try:
        news = yf.Ticker(ticker_symbol).news or []
        headlines = []
        for item in news[:max_headlines]:
            title = item.get("content", {}).get("title") or item.get("title")
            if title:
                headlines.append(title)
        return headlines
    except Exception:
        return []


def get_llm_reasoning(client, row):
    """
    Asks Claude to reason over one stock's metrics + recent headlines:
    why it might be showing up here, and what to watch out for.
    Returns a dict: {"reasoning": str, "flag": str, "flag_reason": str}
    flag is one of: "structural_concern", "macro_or_sector", "unclear"
    Returns None on total failure.
    """
    headlines = get_recent_headlines(row["ticker"])
    headline_block = "\n".join(f"- {h}" for h in headlines) if headlines else "(no recent headlines found)"

    prompt = f"""You are helping a trader sanity-check a stock screen. This stock
passed a rule-based screen: majority-bullish analyst ratings, but the price is
near its 52-week low (a "falling knife vs. buying opportunity" setup).

Ticker: {row['ticker']} ({row.get('company', '')})
Current price: ${row['current_price']}
52-week low: ${row['52wk_low']} ({row['pct_above_low']}% above it)
Analyst ratings: {row['strongBuy']} strong buy, {row['buy']} buy, {row['hold']} hold, \
{row['sell']} sell, {row['strongSell']} strong sell ({row['bullish_pct']}% bullish, \
{row['analyst_count']} analysts)

Recent headlines:
{headline_block}

Classify why this stock is likely near its low, and respond with ONLY a JSON
object (no markdown fences, no preamble) with these exact keys:

"reasoning": 2-3 concise, concrete sentences on why it's near its low despite
  bullish coverage, and what's worth checking before treating this as a buy signal.
"flag": one of "structural_concern", "macro_or_sector", or "unclear".
  Use "structural_concern" ONLY if headlines point to something company-specific
  and fundamental — e.g. accounting/restatement issues, guidance cuts, missed
  earnings, executive departures, regulatory/legal trouble, loss of a major
  customer, debt/liquidity problems. Use "macro_or_sector" if the decline looks
  driven by broad market, sector, or rate-driven selling with no company-specific
  red flag. Use "unclear" if headlines are absent or give no real signal either way.
"flag_reason": one short phrase (under 12 words) naming the specific headline
  detail that drove the flag choice, or "no clear signal in headlines" if unclear.

Be skeptical and concrete, not generic."""

    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            block.text for block in response.content if hasattr(block, "text")
        ).strip()
        text = text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(text)
        return {
            "reasoning": parsed.get("reasoning", "").strip(),
            "flag": parsed.get("flag", "unclear").strip(),
            "flag_reason": parsed.get("flag_reason", "").strip(),
        }
    except Exception as e:
        return {"reasoning": f"(reasoning failed: {e})", "flag": "unclear", "flag_reason": ""}


def add_llm_reasoning(df):
    """Adds an 'llm_reasoning' column to the results DataFrame, in place of a copy."""
    if df.empty:
        return df

    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed (pip install anthropic) — skipping LLM reasoning.")
        return df

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set — skipping LLM reasoning.")
        return df

    client = anthropic.Anthropic(api_key=api_key)

    df = df.copy()
    df["llm_reasoning"] = ""
    df["risk_flag"] = ""
    df["flag_reason"] = ""
    subset = df.head(MAX_STOCKS_TO_REASON_ABOUT)
    for idx, row in subset.iterrows():
        print(f"  Reasoning about {row['ticker']}...")
        result = get_llm_reasoning(client, row)
        df.at[idx, "llm_reasoning"] = result["reasoning"]
        df.at[idx, "risk_flag"] = result["flag"]
        df.at[idx, "flag_reason"] = result["flag_reason"]
        time.sleep(0.5)  # be polite to the API

    return df


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print(f"[{datetime.now()}] Starting screen...")
    tickers = get_sp500_tickers()
    df = run_screen(tickers)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    filename = f"{OUTPUT_DIR}/stock_screen_{timestamp}.csv"

    if df.empty:
        print("No stocks matched the criteria today.")
    else:
        print(f"Found {len(df)} matches.")
        if ENABLE_LLM_REASONING:
            df = add_llm_reasoning(df)
            if "risk_flag" in df.columns:
                # Surface structural concerns first, then macro/sector, then unclear
                flag_order = {"structural_concern": 0, "macro_or_sector": 1, "unclear": 2, "": 3}
                df["_sort"] = df["risk_flag"].map(lambda f: flag_order.get(f, 3))
                df = df.sort_values(["_sort", "pct_above_low"]).drop(columns="_sort")

        df.to_csv(filename, index=False)
        print(f"Saved to {filename}\n")

        # Print a readable summary (full table goes to the CSV)
        FLAG_LABEL = {
            "structural_concern": "⚠️  STRUCTURAL CONCERN",
            "macro_or_sector": "macro/sector-driven",
            "unclear": "unclear signal",
        }
        for _, row in df.iterrows():
            print(f"\n{row['ticker']} ({row.get('company', '')}) — "
                  f"${row['current_price']}, {row['pct_above_low']}% above 52wk low, "
                  f"{row['bullish_pct']}% bullish ({row['analyst_count']} analysts)")
            if row.get("risk_flag"):
                label = FLAG_LABEL.get(row["risk_flag"], row["risk_flag"])
                reason = f" — {row['flag_reason']}" if row.get("flag_reason") else ""
                print(f"  [{label}]{reason}")
            if row.get("llm_reasoning"):
                print(f"  → {row['llm_reasoning']}")


# ----------------------------------------------------------------------
# SCHEDULING (macOS / Linux — cron)
# ----------------------------------------------------------------------
# 1. Find your python3 path:      which python3
# 2. Edit your crontab:           crontab -e
# 3. Add these two lines (adjust paths; cron uses your SYSTEM's local time,
#    so if your machine isn't set to US/Eastern, convert 9:45/10:00 ET first):
#
#    45 9 * * 1-5  cd /path/to/script && /usr/bin/python3 stock_screener.py >> screen.log 2>&1
#    0 10 * * 1-5  cd /path/to/script && /usr/bin/python3 stock_screener.py >> screen.log 2>&1
#
#    (1-5 = Monday-Friday only)
#
# ----------------------------------------------------------------------
# SCHEDULING (Windows — Task Scheduler)
# ----------------------------------------------------------------------
# Create two Basic Tasks that run daily, weekdays only, at 9:45 AM and
# 10:00 AM (in your local time zone — convert from ET if needed), each
# with action: "python.exe" and argument: "C:\path\to\stock_screener.py"
#
# ----------------------------------------------------------------------
# SCHEDULING (cloud — no computer needed)
# ----------------------------------------------------------------------
# GitHub Actions example (.github/workflows/screen.yml), cron is UTC so
# 9:45/10:00 ET becomes 13:45/14:00 UTC during EDT (summer) or 14:45/15:00
# UTC during EST (winter):
#
#   on:
#     schedule:
#       - cron: '45 13 * * 1-5'
#       - cron: '0 14 * * 1-5'
#   jobs:
#     screen:
#       runs-on: ubuntu-latest
#       steps:
#         - uses: actions/checkout@v4
#         - uses: actions/setup-python@v5
#           with: { python-version: '3.11' }
#         - run: pip install yfinance pandas
#         - run: python stock_screener.py
#         - uses: actions/upload-artifact@v4
#           with: { name: screen-results, path: stock_screen_*.csv }
