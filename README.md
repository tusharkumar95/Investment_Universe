# Investment Universe

Investment Universe is the upstream screening engine for Investment Radar.

## What it does

1. Discovers equities across the configured Canadian and Indian exchanges.
2. Removes securities below configurable market-cap, liquidity, and price thresholds.
3. Scores the remaining companies using size, liquidity, valuation, financial health, trend, and data quality.
4. Applies sector and industry concentration limits instead of simply taking the largest companies.
5. Produces exactly the target universe when enough eligible securities exist: 150 Canada + 150 India.
6. Keeps rejected securities and rejection reasons so the screening process remains auditable.

## Outputs

- `data/canada_universe.json`
- `data/india_universe.json`
- `reports/universe_report.json`

## Configuration

Edit `config/universe_rules.json` to change targets, minimums, scoring weights, penalties, and diversification limits.

## Automation

The GitHub Actions workflow can be run manually and is also scheduled on weekdays. Generated universe files are committed back to the repository.

## Architecture

Official exchange listings are the long-term source-of-truth direction. Yahoo Finance via yfinance is currently the discovery/enrichment layer for the first working version. Investment Radar should consume the resulting 300-stock universe and perform the deeper financial, valuation, ownership, and technical analysis.
