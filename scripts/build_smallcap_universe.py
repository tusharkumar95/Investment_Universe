import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

# Deliberately conservative small-cap bands: avoid the weakest microcaps while
# leaving room for companies that can still compound dramatically.
RULES = {
    "Canada": {"region": "ca", "exchanges": ["TOR", "VAN", "NEO", "CNQ"], "min_cap": 250_000_000, "max_cap": 2_000_000_000},
    "India": {"region": "in", "exchanges": ["NSI", "BSE"], "min_cap": 2_000_000_000, "max_cap": 20_000_000_000},
}
TARGET = 10


def num(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def clean(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    return v


def discover(cfg):
    query = yf.EquityQuery("and", [
        yf.EquityQuery("eq", ["region", cfg["region"]]),
        yf.EquityQuery("is-in", ["exchange", *cfg["exchanges"]]),
        yf.EquityQuery("gte", ["intradaymarketcap", cfg["min_cap"]]),
        yf.EquityQuery("lte", ["intradaymarketcap", cfg["max_cap"]]),
    ])
    rows = {}
    offset = 0
    while True:
        result = yf.screen(query, offset=offset, size=250, sortField="intradaymarketcap", sortAsc=False)
        quotes = result.get("quotes", []) if isinstance(result, dict) else []
        if not quotes:
            break
        for q in quotes:
            if q.get("symbol"):
                rows[q["symbol"]] = q
        if len(quotes) < 250:
            break
        offset += 250
        if offset >= 5000:
            break
    return list(rows.values())


def pct_growth(ticker, key):
    return num(ticker.info.get(key))


def score(info, hist):
    revenue = pct_growth(info, "revenueGrowth")
    earnings = pct_growth(info, "earningsGrowth")
    roe = pct_growth(info, "returnOnEquity")
    margin = pct_growth(info, "profitMargins")
    debt = num(info.get("debtToEquity"))
    pe = num(info.get("trailingPE"))
    forward_pe = num(info.get("forwardPE"))
    peg = num(info.get("pegRatio"))
    beta = num(info.get("beta"))
    momentum = 0.0
    if hist is not None and len(hist) >= 126:
        close = hist["Close"].dropna()
        if len(close) >= 126:
            momentum = num(close.iloc[-1] / close.iloc[-126] - 1) or 0.0
    growth_score = np.mean([max(0, min(1, (revenue or 0) / 0.30)), max(0, min(1, (earnings or 0) / 0.40))])
    quality_score = np.mean([max(0, min(1, ((roe or 0) + 0.05) / 0.30)), max(0, min(1, (margin or 0) / 0.25)), max(0, min(1, 1 - (debt or 0) / 250))])
    vals = []
    if pe and pe > 0: vals.append(max(0, min(1, 1 - pe / 60)))
    if forward_pe and forward_pe > 0: vals.append(max(0, min(1, 1 - forward_pe / 50)))
    if peg and peg > 0: vals.append(max(0, min(1, 1 - peg / 3)))
    valuation_score = float(np.mean(vals)) if vals else 0.5
    momentum_score = max(0, min(1, (momentum + 0.30) / 0.90))
    risk_penalty = (0.12 if debt and debt > 250 else 0) + (0.08 if beta and beta > 2.2 else 0) + (0.08 if pe and pe > 100 else 0)
    total = 0.35*growth_score + 0.25*quality_score + 0.15*valuation_score + 0.15*momentum_score + 0.10*0.75 - risk_penalty
    return round(max(0, min(100, total * 100)), 2)


def technical_summary(hist):
    if hist is None or hist.empty:
        return {"historyDays": 0, "sma20": None, "sma50": None, "sma200": None, "momentum3M": None, "momentum6M": None, "momentum1Y": None, "high52Week": None, "low52Week": None, "drawdown52w": None}
    close = hist["Close"].dropna()
    current = num(close.iloc[-1]) if len(close) else None
    def mom(days):
        return num(current / close.iloc[-days] - 1) if current is not None and len(close) > days else None
    high = num(close.tail(252).max()) if len(close) else None
    low = num(close.tail(252).min()) if len(close) else None
    return {"historyDays": int(len(close)), "sma20": num(close.tail(20).mean()), "sma50": num(close.tail(50).mean()) if len(close)>=50 else None, "sma200": num(close.tail(200).mean()) if len(close)>=200 else None, "momentum3M": mom(63), "momentum6M": mom(126), "momentum1Y": mom(252), "high52Week": high, "low52Week": low, "drawdown52w": num(current/high-1) if current is not None and high else None}


def analyze(symbol, row):
    t = yf.Ticker(symbol)
    info = t.info or {}
    hist = t.history(period="5y", auto_adjust=False)
    s = score(info, hist)
    current = num(info.get("currentPrice")) or num(info.get("regularMarketPrice"))
    if current is None and hist is not None and not hist.empty:
        current = num(hist["Close"].dropna().iloc[-1])
    return {
        "ticker": symbol, "symbol": symbol,
        "name": info.get("longName") or info.get("shortName") or row.get("shortName") or symbol,
        "market": "Canada" if symbol.endswith(".TO") or symbol.endswith(".V") else "India",
        "sector": info.get("sector") or row.get("sector") or "Unknown",
        "industry": info.get("industry") or row.get("industry") or "Unknown",
        "marketCap": num(info.get("marketCap")) or num(row.get("marketCap")), "price": current, "currency": info.get("currency"),
        "revenueGrowth": num(info.get("revenueGrowth")), "earningsGrowth": num(info.get("earningsGrowth")), "profitMargin": num(info.get("profitMargins")),
        "roe": num(info.get("returnOnEquity")), "roic": num(info.get("returnOnCapital")), "debtToEquity": num(info.get("debtToEquity")),
        "pe": num(info.get("trailingPE")), "forwardPE": num(info.get("forwardPE")), "peg": num(info.get("pegRatio")), "priceToSales": num(info.get("priceToSalesTrailing12Months")), "priceToBook": num(info.get("priceToBook")), "evToEbitda": num(info.get("enterpriseToEbitda")),
        "freeCashFlow": num(info.get("freeCashflow")), "operatingCashFlow": num(info.get("operatingCashflow")), "insiderOwnership": num(info.get("heldPercentInsiders")), "institutionalOwnership": num(info.get("heldPercentInstitutions")), "dividendYield": num(info.get("dividendYield")), "beta": num(info.get("beta")),
        "universeScore": s,
        "thesis": "High-upside small-cap candidate: growth is weighted most heavily, while business quality, valuation, balance-sheet risk and market confirmation prevent the list from becoming a pure lottery-ticket screen.",
        "screening": {"growth": "Revenue/earnings growth weighted heavily", "quality": "ROE, margin and leverage considered", "valuation": "P/E, forward P/E and PEG where available", "momentum": "3M/6M/1Y price confirmation", "risk": "Debt, beta and extreme valuation penalties"},
        "technical": technical_summary(hist)
    }


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    output = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "method": "Small-cap asymmetric-upside screen", "markets": {}}
    for market, cfg in RULES.items():
        print(f"Discovering small caps: {market}")
        candidates = discover(cfg)
        print(f"  candidates: {len(candidates)}")
        analyzed = []
        for row in candidates[:150]:
            try: analyzed.append(analyze(row["symbol"], row))
            except Exception as exc: print(f"  skip {row.get('symbol')}: {exc}")
        analyzed.sort(key=lambda x: x.get("universeScore", 0), reverse=True)
        selected = analyzed[:TARGET]
        output["markets"][market] = {"target": TARGET, "selected_count": len(selected), "stocks": selected, "candidate_count": len(candidates)}
        if len(selected) < TARGET: raise RuntimeError(f"{market}: only {len(selected)} usable small-cap candidates")
    with open(os.path.join(DATA_DIR, "smallcap_universe.json"), "w", encoding="utf-8") as f: json.dump(output, f, indent=2, ensure_ascii=False)
    print(json.dumps({m: output["markets"][m]["selected_count"] for m in output["markets"]}, indent=2))


if __name__ == "__main__": main()
