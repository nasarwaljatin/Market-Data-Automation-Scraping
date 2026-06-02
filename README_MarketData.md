# Market Data Automation & Scraping

A Python-based data pipeline that integrates, standardizes, and preprocesses market data from multiple financial sources — **NSE**, **Yahoo Finance**, and **Google Finance** — to power risk management models and parameter optimization workflows.

---

## Overview

Financial data sourced from different providers is often inconsistent in format, timezone, naming conventions, and precision. This project automates the full pipeline from raw data ingestion to clean, analysis-ready datasets — eliminating manual effort and improving the reliability of downstream risk models.

**Key Outcomes:**
- 📈 **70% improvement** in data reliability by resolving cross-source inconsistencies
- ⚡ **70–80% reduction** in manual preprocessing effort for risk management workflows
- 📊 **35% improvement in Sharpe ratio** through better data quality and feature engineering

---

## Features

- **Multi-source ingestion** — fetches data from NSE (via scraping), Yahoo Finance (`yfinance`), and Google Finance
- **Data standardization** — aligns column names, date formats, OHLCV schema, and timezone across all sources
- **Conflict resolution** — detects and resolves discrepancies when the same instrument has differing values across sources
- **Automated preprocessing** — handles missing values, outlier detection, corporate action adjustments, and resampling
- **Feature engineering** — generates financial features (rolling returns, volatility, momentum indicators) used by risk models
- **Pipeline scheduling** — supports automated daily runs via cron or a task scheduler

---

## Project Structure

```
market_data/
├── scrapers/
│   ├── nse_scraper.py        # NSE data fetcher (requests + BeautifulSoup / nsepython)
│   ├── yahoo_fetcher.py      # Yahoo Finance data via yfinance
│   └── google_fetcher.py     # Google Finance data fetcher
├── pipeline/
│   ├── standardizer.py       # Schema alignment and normalization across sources
│   ├── conflict_resolver.py  # Cross-source discrepancy detection and resolution
│   ├── preprocessor.py       # Missing values, outliers, corporate actions
│   └── feature_engineer.py  # Rolling features, volatility, momentum indicators
├── models/
│   └── risk_model.py         # Risk management model integration
├── utils/
│   ├── logger.py             # Structured logging
│   └── config.py             # Source URLs, symbols, date ranges
├── data/
│   ├── raw/                  # Raw data fetched from each source
│   └── processed/            # Clean, merged, analysis-ready datasets
├── notebooks/
│   └── exploratory.ipynb     # EDA and data quality analysis
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/market-data-automation.git
cd market-data-automation
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Data Sources

Edit `utils/config.py` to set your target symbols, date ranges, and any API credentials:

```python
SYMBOLS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
START_DATE = "2020-01-01"
END_DATE   = "2024-12-31"
```

---

## How to Run

### Run the Full Pipeline

```bash
python pipeline/run_pipeline.py
```

This will:
1. Fetch data from NSE, Yahoo Finance, and Google Finance
2. Standardize schemas and resolve cross-source conflicts
3. Preprocess and clean the merged dataset
4. Engineer features for risk model input
5. Save outputs to `data/processed/`

### Run Individual Stages

```bash
# Fetch raw data only
python scrapers/nse_scraper.py --symbol RELIANCE --start 2023-01-01 --end 2024-01-01

# Standardize and merge
python pipeline/standardizer.py

# Generate features
python pipeline/feature_engineer.py
```

---

## Data Sources

| Source         | Method                        | Data Provided                        |
|----------------|-------------------------------|--------------------------------------|
| NSE            | Web scraping / `nsepython`    | OHLCV, delivery data, index data     |
| Yahoo Finance  | `yfinance` library            | OHLCV, adjusted close, dividends     |
| Google Finance | HTTP requests + parsing       | OHLCV, real-time snapshots           |

---

## Preprocessing Steps

1. **Schema alignment** — rename columns to a unified `open`, `high`, `low`, `close`, `volume` format
2. **Timezone normalization** — convert all timestamps to IST (UTC+5:30)
3. **Missing value handling** — forward-fill for minor gaps; flag and log large gaps
4. **Outlier detection** — Z-score and IQR-based flagging with configurable thresholds
5. **Corporate action adjustment** — split and dividend adjustments applied to historical prices
6. **Conflict resolution** — when sources disagree beyond a threshold, a weighted priority rule is applied (NSE > Yahoo > Google for Indian equities)

---

## Feature Engineering

Features generated for risk model input:

| Feature                  | Description                                      |
|--------------------------|--------------------------------------------------|
| `log_return`             | Daily log returns                                |
| `rolling_volatility_20`  | 20-day rolling standard deviation of returns     |
| `momentum_14`            | 14-day price momentum                            |
| `rsi_14`                 | 14-period Relative Strength Index                |
| `volume_zscore`          | Z-score normalized volume                        |
| `beta_rolling`           | Rolling beta against benchmark (e.g., Nifty 50)  |

---

## Results

| Metric                        | Before  | After   | Improvement |
|-------------------------------|---------|---------|-------------|
| Data reliability score        | Baseline| +70%    | ✅           |
| Manual preprocessing effort   | ~8 hrs/day | ~1.5 hrs/day | −75% ✅ |
| Risk model Sharpe ratio       | Baseline| +35%    | ✅           |

---

## Requirements

```
yfinance>=0.2.38
pandas>=2.0.0
numpy>=1.26.0
requests>=2.31.0
beautifulsoup4>=4.12.0
nsepython>=2.0.0
scikit-learn>=1.4.0
schedule>=1.2.0
python-dotenv>=1.0.0
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## Logging

All pipeline stages log to `logs/pipeline.log` with timestamps, source names, record counts, and any data quality warnings or errors encountered during ingestion and processing.

---

## Assumptions

- Primary market is **Indian equities (NSE)**; Yahoo and Google Finance serve as supplementary and validation sources.
- All timestamps are normalized to **IST (UTC+5:30)**.
- For conflicting OHLCV values across sources, **NSE data takes priority** for Indian instruments.
- The pipeline is designed for **end-of-day (EOD) data**; intraday support is not included.
- Risk model integration expects features in a flat CSV format as produced by `feature_engineer.py`.

---

## License

This project is intended for research and portfolio demonstration purposes only. Not financial advice.
