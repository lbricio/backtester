"""Métricas de performance."""

from __future__ import annotations

import numpy as np
import pandas as pd

# chave -> (rótulo, formato). Formatos: pct, num, int, money
METRIC_INFO: dict[str, tuple[str, str]] = {
    "final_equity": ("Patrimônio final", "money"),
    "total_return": ("Retorno total", "pct"),
    "cagr": ("CAGR", "pct"),
    "volatility": ("Volatilidade anual", "pct"),
    "sharpe": ("Sharpe", "num"),
    "sortino": ("Sortino", "num"),
    "max_drawdown": ("Drawdown máximo", "pct"),
    "max_dd_bars": ("Duração máx. drawdown (candles)", "int"),
    "calmar": ("Calmar", "num"),
    "n_trades": ("Nº de operações", "int"),
    "win_rate": ("Taxa de acerto", "pct"),
    "profit_factor": ("Profit factor", "num"),
    "avg_trade": ("Retorno médio por operação", "pct"),
    "best_trade": ("Melhor operação", "pct"),
    "worst_trade": ("Pior operação", "pct"),
    "avg_bars": ("Duração média (candles)", "num"),
    "exposure": ("Tempo posicionado", "pct"),
}

# métricas em que "maior é melhor" úteis para otimização
OPTIMIZABLE = ["sharpe", "sortino", "cagr", "total_return", "calmar", "profit_factor", "max_drawdown"]


def periods_per_year(index: pd.Index) -> float:
    """Estima o número de candles por ano a partir do espaçamento do índice."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 3:
        return 252.0
    delta = pd.Series(index).diff().dropna().median().total_seconds()
    if delta <= 0:
        return 252.0
    day = 86400.0
    if delta >= 25 * day:
        return 12.0
    if delta >= 5 * day:
        return 52.0
    if delta >= 0.8 * day:
        # diário: conta os candles por ano de fato (B3 ≈ 252, câmbio ≈ 260, cripto ≈ 365)
        span_years = (index[-1] - index[0]).total_seconds() / (365.25 * day)
        if span_years >= 0.5:
            return float(np.clip((len(index) - 1) / span_years, 200, 366))
        return 365.0 if pd.Series(index.day_of_week).isin([5, 6]).mean() > 0.2 else 252.0
    # intradiário: assume pregão de 6,5h
    return 252.0 * (6.5 * 3600.0) / delta


def drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


def _max_dd_duration(dd: pd.Series) -> int:
    longest = cur = 0
    for v in dd.to_numpy():
        cur = cur + 1 if v < 0 else 0
        longest = max(longest, cur)
    return longest


def compute_metrics(
    equity: pd.Series,
    ppy: float,
    trades: pd.DataFrame | None = None,
    exposure: pd.Series | None = None,
    risk_free: float = 0.0,
) -> dict:
    """Métricas de uma curva de patrimônio.

    risk_free: taxa livre de risco anual (ex.: 0.10 = 10% a.a., como o CDI). Sharpe e Sortino
    usam o retorno em excesso a ela; com 0, medem o retorno bruto.
    """
    m: dict[str, float] = {}
    eq = equity.astype(float)
    rets = eq.pct_change().dropna()
    n = len(eq)

    m["final_equity"] = float(eq.iloc[-1])
    m["total_return"] = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = max((n - 1) / ppy, 1e-9)
    ratio = eq.iloc[-1] / eq.iloc[0]
    m["cagr"] = float(ratio ** (1.0 / years) - 1.0) if ratio > 0 else -1.0

    rf = (1.0 + risk_free) ** (1.0 / ppy) - 1.0  # taxa por período
    excess = rets - rf
    std = rets.std(ddof=1) if len(rets) > 1 else 0.0
    m["volatility"] = float(std * np.sqrt(ppy))
    m["sharpe"] = float(excess.mean() / std * np.sqrt(ppy)) if std > 0 else 0.0
    downside = np.sqrt((np.minimum(excess, 0.0) ** 2).mean()) if len(rets) else 0.0
    m["sortino"] = float(excess.mean() / downside * np.sqrt(ppy)) if downside > 0 else 0.0

    dd = drawdown(eq)
    m["max_drawdown"] = float(dd.min())
    m["max_dd_bars"] = _max_dd_duration(dd)
    m["calmar"] = float(m["cagr"] / abs(m["max_drawdown"])) if m["max_drawdown"] < 0 else 0.0

    if trades is not None:
        m["n_trades"] = int(len(trades))
        if len(trades):
            r = trades["return_pct"].astype(float)
            pnl = trades["pnl"].astype(float)
            gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
            m["win_rate"] = float((pnl > 0).mean())
            m["profit_factor"] = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
            m["avg_trade"] = float(r.mean())
            m["best_trade"] = float(r.max())
            m["worst_trade"] = float(r.min())
            m["avg_bars"] = float(trades["bars"].mean())
        else:
            for k in ("win_rate", "profit_factor", "avg_trade", "best_trade", "worst_trade", "avg_bars"):
                m[k] = 0.0
    if exposure is not None:
        m["exposure"] = float((exposure.abs() > 1e-9).mean())
    return m


def format_metric(key: str, value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    fmt = METRIC_INFO.get(key, ("", "num"))[1]
    if isinstance(value, float) and np.isinf(value):
        return "∞"
    if fmt == "pct":
        return f"{value * 100:,.2f}%"
    if fmt == "int":
        return f"{int(value):,}"
    if fmt == "money":
        return f"{value:,.2f}"
    return f"{value:,.3f}"
