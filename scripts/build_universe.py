import json
import math
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config", "universe_rules.json")
DATA_DIR = os.path.join(ROOT, "data")
REPORT_DIR = os.path.join(ROOT, "reports")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def num(value):
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def clean(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value) if not isinstance(value, (list, dict, tuple)) else False:
        return None
    return value


def discover_market(region, exchanges):
    rows = {}
    for exchange in exchanges:
        print(f"Discovering {region.upper()} / {exchange} ...")
        try:
            query = yf.EquityQuery(
                "and",
                [
                    yf.EquityQuery("eq", ["region", region]),
                    yf.EquityQuery("eq", ["exchange", exchange]),
                    yf.EquityQuery("eq", ["quoteType", "EQUITY"]),
                ],
            )
            offset = 0
            while True:
                result = yf.screen(
                    query,
                    offset=offset,
                    size=250,
                    sortField="marketcap",
                    sortAsc=False,
                )
                quotes = result.get("quotes", []) if isinstance(result, dict) else []
                if not quotes:
                    break
                for quote in quotes:
                    symbol = quote.get("symbol")
                    if symbol:
                        quote["discovered_exchange"] = exchange
                        rows[symbol] = quote
                print(f"  +{len(quotes)} (total unique: {len(rows)})")
                if len(quotes) < 250:
                    break
                offset += 250
                if offset >= 10000:
                    print("  Reached Yahoo screener safety limit of 10,000 rows.")
                    break
        except Exception as exc:
            print(f"  WARNING: discovery failed for {exchange}: {exc}")
    return list(rows.values())


def eligibility(row, rules):
    e = rules["eligibility"]
    market_cap = num(row.get("marketCap"))
    volume = num(row.get("averageDailyVolume3Month"))
    price = num(row.get("regularMarketPrice"))
    reasons = []

    if e.get("require_market_cap") and market_cap is None:
        reasons.append("missing_market_cap")
    elif market_cap is not None and market_cap < e["min_market_cap"]:
        reasons.append("market_cap_below_minimum")

    if e.get("require_volume") and volume is None:
        reasons.append("missing_3m_volume")
    elif volume is not None and volume < e["min_avg_daily_volume_3m"]:
        reasons.append("volume_below_minimum")

    if e.get("require_price") and price is None:
        reasons.append("missing_price")
    elif price is not None and price < e["min_price"]:
        reasons.append("price_below_minimum")

    return reasons


def percentile(values, value, reverse=False):
    clean_values = [v for v in values if v is not None and math.isfinite(v)]
    if value is None or not clean_values:
        return 0.0
    rank = sum(v <= value for v in clean_values) / len(clean_values)
    return rank if not reverse else 1.0 - rank


def score_rows(rows, rules):
    metrics = {
        "market_cap": [num(r.get("marketCap")) for r in rows],
        "volume": [num(r.get("averageDailyVolume3Month")) for r in rows],
        "pe": [num(r.get("trailingPE")) for r in rows],
        "pb": [num(r.get("priceToBook")) for r in rows],
        "roe": [num(r.get("returnOnEquity")) for r in rows],
        "debt": [num(r.get("debtToEquity")) for r in rows],
        "change": [num(r.get("regularMarketChangePercent")) for r in rows],
    }
    w = rules["ranking"]
    q = rules["quality"]

    for row in rows:
        mc = num(row.get("marketCap"))
        vol = num(row.get("averageDailyVolume3Month"))
        pe = num(row.get("trailingPE"))
        pb = num(row.get("priceToBook"))
        roe = num(row.get("returnOnEquity"))
        debt = num(row.get("debtToEquity"))
        change = num(row.get("regularMarketChangePercent"))

        mc_score = percentile(metrics["market_cap"], mc)
        liq_score = percentile(metrics["volume"], vol)

        valuation_parts = []
        if pe is not None and pe > 0:
            valuation_parts.append(percentile(metrics["pe"], pe, reverse=True))
        if pb is not None and pb > 0:
            valuation_parts.append(percentile(metrics["pb"], pb, reverse=True))
        valuation_score = sum(valuation_parts) / len(valuation_parts) if valuation_parts else 0.5

        health_parts = []
        if roe is not None:
            health_parts.append(max(0.0, min(1.0, (roe + 0.10) / 0.35)))
        if debt is not None:
            health_parts.append(max(0.0, min(1.0, 1.0 - debt / 200.0)))
        health_score = sum(health_parts) / len(health_parts) if health_parts else 0.5

        trend_score = percentile(metrics["change"], change)
        quality_score = 1.0
        missing = sum(x is None for x in [mc, vol, pe, pb, roe, debt, change])
        quality_score = max(0.0, 1.0 - missing / 7.0)

        raw = (
            w["market_cap"] * mc_score
            + w["liquidity"] * liq_score
            + w["valuation"] * valuation_score
            + w["financial_health"] * health_score
            + w["ownership"] * 0.5
            + w["trend"] * trend_score
            + w["data_quality"] * quality_score
        )

        penalty = 0.0
        if pe is not None and pe <= 0:
            penalty += q["negative_pe_penalty"]
        elif pe is not None and pe > 80:
            penalty += q["extreme_pe_penalty"]
        if debt is not None and debt > 250:
            penalty += q["high_debt_penalty"]
        if change is not None and change < -10:
            penalty += q["negative_change_penalty"]

        row["universe_score"] = round(max(0.0, raw - penalty), 3)
        row["screening_penalty"] = round(penalty, 3)

    return rows


def select_diversified(rows, target, rules):
    rows = sorted(rows, key=lambda r: r["universe_score"], reverse=True)
    max_sector = max(1, math.floor(target * rules["diversification"]["max_sector_share"]))
    max_industry = max(1, math.floor(target * rules["diversification"]["max_industry_share"]))

    selected = []
    sector_counts = {}
    industry_counts = {}
    rejected = []

    for row in rows:
        sector = row.get("sector") or "Unknown"
        industry = row.get("industry") or "Unknown"
        if sector_counts.get(sector, 0) >= max_sector:
            rejected.append((row, "sector_concentration"))
            continue
        if industry_counts.get(industry, 0) >= max_industry:
            rejected.append((row, "industry_concentration"))
            continue
        selected.append(row)
        sector_counts[sector] = sector_counts.get(sector, 0) + 1
        industry_counts[industry] = industry_counts.get(industry, 0) + 1
        if len(selected) >= target:
            break

    if len(selected) < target:
        selected_ids = {r.get("symbol") for r in selected}
        for row in rows:
            if len(selected) >= target:
                break
            if row.get("symbol") in selected_ids:
                continue
            selected.append(row)
            selected_ids.add(row.get("symbol"))
            sector_counts[row.get("sector") or "Unknown"] = sector_counts.get(row.get("sector") or "Unknown", 0) + 1
            industry_counts[row.get("industry") or "Unknown"] = industry_counts.get(row.get("industry") or "Unknown", 0) + 1

    selected_symbols = {r.get("symbol") for r in selected}
    for row, reason in rejected:
        if row.get("symbol") not in selected_symbols:
            row["rejection_reason"] = reason

    return selected, sector_counts, industry_counts


def simplify(row):
    fields = [
        "symbol", "shortName", "exchange", "discovered_exchange", "quoteType",
        "sector", "industry", "marketCap", "averageDailyVolume3Month",
        "regularMarketPrice", "regularMarketChangePercent", "trailingPE",
        "forwardPE", "priceToBook", "returnOnEquity", "debtToEquity",
        "universe_score", "screening_penalty", "rejection_reason"
    ]
    return {k: clean(row.get(k)) for k in fields if row.get(k) is not None}


def build_market(name, cfg, rules):
    discovered = discover_market(cfg["region"], cfg["exchanges"])
    print(f"{name}: discovered {len(discovered)} unique equities")

    eligible = []
    rejected = []
    for row in discovered:
        reasons = eligibility(row, rules)
        if reasons:
            row["rejection_reason"] = ";".join(reasons)
            rejected.append(row)
        else:
            eligible.append(row)

    print(f"{name}: eligible {len(eligible)} / rejected {len(rejected)}")
    scored = score_rows(eligible, rules)
    selected, sectors, industries = select_diversified(
        scored, rules["target_count"][name], rules
    )

    selected_symbols = {r["symbol"] for r in selected}
    final_rejected = []
    for row in rejected + scored:
        if row.get("symbol") not in selected_symbols:
            final_rejected.append(simplify(row))

    output = {
        "market": name,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance screener via yfinance; exchange/region filters",
        "target_count": rules["target_count"][name],
        "discovered_count": len(discovered),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "stocks": [simplify(r) for r in selected],
        "rejected": final_rejected,
        "sector_counts": dict(sorted(sectors.items(), key=lambda x: (-x[1], x[0]))),
        "industry_counts": dict(sorted(industries.items(), key=lambda x: (-x[1], x[0]))),
    }
    return output


def main():
    rules = load_config()
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    outputs = {}
    for market in ("Canada", "India"):
        outputs[market] = build_market(
            market, rules["discovery"][market], rules
        )
        filename = "canada_universe.json" if market == "Canada" else "india_universe.json"
        with open(os.path.join(DATA_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(outputs[market], f, indent=2, ensure_ascii=False)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_version": rules.get("version"),
        "markets": {
            market: {
                "discovered": outputs[market]["discovered_count"],
                "eligible": outputs[market]["eligible_count"],
                "selected": outputs[market]["selected_count"],
                "target": outputs[market]["target_count"],
            }
            for market in outputs
        },
        "status": "PASS" if all(
            outputs[m]["selected_count"] >= rules["target_count"][m]
            for m in outputs
        ) else "PARTIAL",
    }
    with open(os.path.join(REPORT_DIR, "universe_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== INVESTMENT UNIVERSE BUILD COMPLETE ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
