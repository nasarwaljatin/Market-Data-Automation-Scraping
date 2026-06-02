"""
models/risk_model.py
---------------------
Risk management model that consumes the feature-engineered flat CSV output
from feature_engineer.py and computes portfolio-level risk metrics.

Metrics computed:
  - Sharpe Ratio          : annualised risk-adjusted return
  - Sortino Ratio         : downside deviation adjusted return
  - Max Drawdown          : largest peak-to-trough decline
  - VaR (95%, 99%)        : Value at Risk (parametric + historical)
  - CVaR / Expected Shortfall : tail risk beyond VaR
  - Annualised Volatility
  - Calmar Ratio          : CAGR / Max Drawdown

Standalone:
    python models/risk_model.py
    python models/risk_model.py --symbol RELIANCE.NS
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import PROC_DATA_DIR, SYMBOLS
from utils.logger import get_logger

log = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE        = 0.065   # 6.5% annualised (approx. RBI repo rate)


# ──────────────────────────────────────────────────────────────────────────────
# Individual metric functions
# ──────────────────────────────────────────────────────────────────────────────

def sharpe_ratio(returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    """
    Annualised Sharpe Ratio.
    SR = (mean_daily_return - rf_daily) / std_daily × √252
    """
    if returns.empty or returns.std() == 0:
        return np.nan
    rf_daily   = risk_free / TRADING_DAYS_PER_YEAR
    excess_ret = returns - rf_daily
    return (excess_ret.mean() / excess_ret.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)


def sortino_ratio(returns: pd.Series, risk_free: float = RISK_FREE_RATE) -> float:
    """
    Annualised Sortino Ratio using downside deviation.
    """
    if returns.empty:
        return np.nan
    rf_daily     = risk_free / TRADING_DAYS_PER_YEAR
    excess_ret   = returns - rf_daily
    downside_ret = excess_ret[excess_ret < 0]
    if downside_ret.empty or downside_ret.std() == 0:
        return np.nan
    downside_std = np.sqrt((downside_ret ** 2).mean())
    return (excess_ret.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown(cum_returns: pd.Series) -> float:
    """
    Maximum peak-to-trough drawdown of a cumulative return series.

    Parameters
    ----------
    cum_returns : pd.Series — cumulative product of (1 + daily_return)

    Returns
    -------
    float — max drawdown as a negative fraction, e.g. -0.35 = -35%
    """
    rolling_max = cum_returns.cummax()
    drawdowns   = (cum_returns - rolling_max) / rolling_max
    return float(drawdowns.min())


def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical Value at Risk (negative number = loss)."""
    if returns.empty:
        return np.nan
    return float(np.percentile(returns.dropna(), (1 - confidence) * 100))


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """Parametric (Gaussian) VaR."""
    if returns.empty:
        return np.nan
    mu  = returns.mean()
    sig = returns.std()
    return float(stats.norm.ppf(1 - confidence, mu, sig))


def cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """Conditional VaR / Expected Shortfall (average loss beyond VaR threshold)."""
    if returns.empty:
        return np.nan
    threshold = var_historical(returns, confidence)
    tail = returns[returns <= threshold]
    return float(tail.mean()) if not tail.empty else np.nan


def annualised_volatility(returns: pd.Series) -> float:
    """Annualised volatility (standard deviation × √252)."""
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def calmar_ratio(returns: pd.Series) -> float:
    """
    Calmar Ratio = CAGR / |Max Drawdown|
    CAGR estimated from daily log return sum.
    """
    if returns.empty:
        return np.nan
    cum = (1 + returns).cumprod()
    years = len(returns) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        return np.nan
    cagr = (cum.iloc[-1] ** (1 / years)) - 1
    mdd  = abs(max_drawdown(cum))
    return float(cagr / mdd) if mdd != 0 else np.nan


# ──────────────────────────────────────────────────────────────────────────────
# Full risk report
# ──────────────────────────────────────────────────────────────────────────────

def compute_risk_report(df: pd.DataFrame, symbol: str) -> dict:
    """
    Compute all risk metrics for a single instrument.

    Parameters
    ----------
    df     : pd.DataFrame — must contain 'log_return' column
    symbol : str

    Returns
    -------
    dict — metric name → value
    """
    if "log_return" not in df.columns:
        log.error(f"[RiskModel] 'log_return' column missing for {symbol}")
        return {}

    returns = df["log_return"].dropna()
    cum_ret = (1 + returns).cumprod()

    report = {
        "symbol":                 symbol,
        "n_trading_days":         len(returns),
        "mean_daily_return":      float(returns.mean()),
        "annualised_return_pct":  float(returns.mean() * TRADING_DAYS_PER_YEAR * 100),
        "annualised_volatility":  annualised_volatility(returns),
        "sharpe_ratio":           sharpe_ratio(returns),
        "sortino_ratio":          sortino_ratio(returns),
        "max_drawdown_pct":       max_drawdown(cum_ret) * 100,
        "calmar_ratio":           calmar_ratio(returns),
        "var_95_historical":      var_historical(returns, 0.95),
        "var_99_historical":      var_historical(returns, 0.99),
        "var_95_parametric":      var_parametric(returns, 0.95),
        "cvar_95":                cvar(returns, 0.95),
        "cvar_99":                cvar(returns, 0.99),
        "skewness":               float(returns.skew()),
        "kurtosis":               float(returns.kurtosis()),
    }

    log.info(
        f"[RiskModel] {symbol} | "
        f"Sharpe={report['sharpe_ratio']:.2f} | "
        f"MaxDD={report['max_drawdown_pct']:.1f}% | "
        f"Vol={report['annualised_volatility']:.2%}"
    )
    return report


def print_report(report: dict) -> None:
    """Pretty-print a risk report dict."""
    if not report:
        return
    print(f"\n{'═' * 55}")
    print(f"  Risk Report: {report.get('symbol', 'N/A')}")
    print(f"{'═' * 55}")
    for k, v in report.items():
        if k == "symbol":
            continue
        label = k.replace("_", " ").title()
        val   = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {label:<35} {val:>12}")
    print(f"{'═' * 55}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Load helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_features(symbol: str) -> pd.DataFrame:
    safe_sym = symbol.replace(".", "_")
    path = os.path.join(PROC_DATA_DIR, f"features_{safe_sym}.csv")
    if not os.path.exists(path):
        log.warning(f"[RiskModel] Features file not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["date"])
    log.debug(f"[RiskModel] Loaded {path} ({len(df)} rows)")
    return df


def save_risk_report(report: dict, symbol: str) -> str:
    safe_sym = symbol.replace(".", "_")
    filename = f"risk_report_{safe_sym}.csv"
    filepath = os.path.join(PROC_DATA_DIR, filename)
    pd.DataFrame([report]).to_csv(filepath, index=False)
    log.info(f"[RiskModel] Saved → {filepath}")
    return filepath


def run_risk_model(symbols: list = None) -> list:
    """Run risk model for all symbols and return list of reports."""
    if symbols is None:
        symbols = SYMBOLS

    reports = []
    for symbol in symbols:
        df = load_features(symbol)
        if df.empty:
            log.warning(f"[RiskModel] Skipping {symbol} — no feature data")
            continue
        report = compute_risk_report(df, symbol)
        if report:
            reports.append(report)
            save_risk_report(report, symbol)

    return reports


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Risk model — compute Sharpe, VaR, drawdown, etc.")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to analyse")
    args = parser.parse_args()

    log.info("=== RiskModel: starting ===")
    reports = run_risk_model(args.symbols)
    for r in reports:
        print_report(r)
    log.info("=== RiskModel: complete ===")
