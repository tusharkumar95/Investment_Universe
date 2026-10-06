import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

RULES = {
    "Canada": {"region": "ca", "exchanges": ["TOR", "VAN", "NEO", "CNQ"], "min_cap": 250_000_000, "max_cap": 2_000_000_000},
    "India": {"region": "in", "exchanges": ["NSI", "BSE"], "min_cap": 2_000_000_000, "max_cap": 20_000_000_000},
}
TARGET = 10
ANALYZE_LIMIT = 300
MAX_PER_INDUSTRY = 3
EXCLUDED_TYPES = ("split corp", "split share", "closed-end", "closed end", "investment trust", "acquisition corp", "capital pool")


def num(v):
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def business_fit(info, row):
    text = " ".join(str(x or "") for x in [
        info.get("longName"), info.get("shortName"), info.get("industry"),
        info.get("quoteType"), row.get("shortName"), row.get("industry")
    ]).lower()
    return not any(term in text for term in EXCLUDED_TYPES)


def data_confidence(info):
    fields = [
        "revenueGrowth", "earningsGrowth", "returnOnEquity", "profitMargins",
        "debtToEquity", "trailingPE", "forwardPE", "freeCashflow",
        "operatingCashflow", "marketCap"
    ]
    present = sum(num(info.get(k)) is not None for k in fields)
    return present / len(fields)


def score(info, hist):
    revenue, earnings = num(info.get("revenueGrowth")), num(info.get("earningsGrowth"))
    roe, margin, debt = num(info.get("returnOnEquity")), num(info.get("profitMargins")), num(info.get("debtToEquity"))
    pe, forward_pe, peg, beta = num(info.get("trailingPE")), num(info.get("forwardPE")), num(info.get("pegRatio")), num(info.get("beta"))
    fcf, ocf, confidence = num(info.get("freeCashflow")), num(info.get("operatingCashflow")), data_confidence(info)
    momentum = 0.0
    if hist is not None and not hist.empty:
        close = hist["Close"].dropna()
        if len(close) >= 126:
            momentum = num(close.iloc[-1] / close.iloc[-126] - 1) or 0.0
    growth_vals = [max(0, min(1, x)) for x in [((revenue or 0)+0.05)/0.35, ((earnings or 0)+0.05)/0.45]]
    growth_score = float(np.mean(growth_vals))
    quality_parts = [
        max(0, min(1, ((roe or 0)+0.05)/0.30)),
        max(0, min(1, ((margin or 0)+0.05)/0.30)),
        max(0, min(1, 1-(debt or 0)/250))
    ]
    if fcf is not None: quality_parts.append(1.0 if fcf > 0 else 0.0)
    if ocf is not None: quality_parts.append(1.0 if ocf > 0 else 0.0)
    quality_score = float(np.mean(quality_parts))
    vals = []
    if pe and pe > 0: vals.append(max(0, min(1, 1-pe/60)))
    if forward_pe and forward_pe > 0: vals.append(max(0, min(1, 1-forward_pe/50)))
    if peg and peg > 0: vals.append(max(0, min(1, 1-peg/3)))
    valuation_score = float(np.mean(vals)) if vals else 0.30
    momentum_score = max(0, min(1, (momentum+0.30)/0.90))
    risk_penalty = 0
    if debt is not None and debt > 250: risk_penalty += 0.12
    if beta is not None and beta > 2.2: risk_penalty += 0.08
    if pe is not None and pe > 100: risk_penalty += 0.08
    if fcf is not None and fcf < 0: risk_penalty += 0.07
    if ocf is not None and ocf < 0: risk_penalty += 0.08
    total = 0.30*growth_score + 0.30*quality_score + 0.15*valuation_score + 0.10*momentum_score + 0.15*confidence - risk_penalty
    result = max(0, min(100, total*100))
    if confidence < 0.50: result = min(result, 64)
    return round(result, 2), round(confidence*100, 1)


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


def analyze(symbol, row):
    ticker = yf.Ticker(symbol)
    info = ticker.info or {}
    hist = ticker.history(period="5y", auto_adjust=False)
    if not business_fit(info, row):
        return None
    s, confidence = score(info, hist)
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
        "universeScore": s, "dataConfidence": confidence, "businessFit": "Operating company",
        "thesis": "Small-cap operating-company candidate screened for growth, quality, valuation, cash-flow health, leverage, data confidence and market confirmation.",
        "screening": {"growth": "Revenue and earnings growth", "quality": "ROE, margin, cash flow and leverage", "valuation": "P/E, forward P/E and PEG where available", "momentum": "6M market confirmation", "confidence": "10-field fundamental coverage", "risk": "Debt, beta, negative cash flow and extreme valuation penalties"},
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
        for row in candidates[:ANALYZE_LIMIT]:
            try:
                result = analyze(row["symbol"], row)
                if result: analyzed.append(result)
            except Exception as exc: print(f"  skip {row.get('symbol')}: {exc}")
        analyzed.sort(key=lambda x: x.get("universeScore", 0), reverse=True)
        selected, industry_counts, company_keys = [], {}, set()
        for stock in analyzed:
            industry = stock.get("industry") or "Unknown"
            company_key = str(stock.get("ticker") or "").upper().removesuffix(".NS").removesuffix(".BO")
            if company_key in company_keys:
                continue
            if industry_counts.get(industry, 0) >= MAX_PER_INDUSTRY:
                continue
            selected.append(stock)
            company_keys.add(company_key)
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
            if len(selected) >= TARGET:
                break
        output["markets"][market] = {"target": TARGET, "selected_count": len(selected), "stocks": selected, "candidate_count": len(candidates), "analyzed_count": len(analyzed), "max_per_industry": MAX_PER_INDUSTRY}
        if len(selected) < TARGET: raise RuntimeError(f"{market}: only {len(selected)} usable small-cap candidates")
    with open(os.path.join(DATA_DIR, "smallcap_universe.json"), "w", encoding="utf-8") as f: json.dump(output, f, indent=2, ensure_ascii=False)
    print(json.dumps({m: output["markets"][m]["selected_count"] for m in output["markets"]}, indent=2))


if __name__ == "__main__": main()
