"""Varredura: várias estratégias × vários ativos.

Cada combinação roda um backtest completo com os mesmos custos e regras. O resultado é
comparado com dois referenciais nas MESMAS datas do backtest:
- o CDI que realmente rendeu no período (Banco Central);
- o Buy & Hold do próprio ativo.

Os resumos priorizam robustez: em quantos % dos ativos a estratégia ganha, e o valor
mediano — não o melhor caso, que costuma ser sorte.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from quantbt.engine import BacktestConfig, run_backtest
from quantbt.metrics import periods_per_year
from quantbt.strategies import Strategy

RESULT_COLUMNS = [
    "ticker", "strategy", "sharpe", "cagr", "cdi_cagr", "excess_cdi", "bh_cagr", "excess_bh",
    "total_return", "max_drawdown", "n_trades", "trades_per_year", "exposure", "years", "error",
]

# métricas que dá para mostrar no mapa de calor: chave -> (rótulo, formato, centro da escala de cores)
HEATMAP_METRICS = {
    "excess_cdi": ("Ganho sobre o CDI (a.a.)", "pct", 0.0),
    "excess_bh": ("Ganho sobre o Buy & Hold (a.a.)", "pct", 0.0),
    "sharpe": ("Sharpe", "num", 0.0),
    "cagr": ("CAGR", "pct", None),
    "max_drawdown": ("Drawdown máximo", "pct", None),
}


def run_scan(
    data: dict[str, pd.DataFrame],
    strategies: list[Strategy],
    config: BacktestConfig,
    params: dict[str, dict] | None = None,
    cdi: pd.Series | None = None,
    min_bars: int = 60,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Roda todas as estratégias em todos os ativos.

    params: parâmetros por nome de estratégia (sem entrada = padrões da estratégia).
    cdi: índice acumulado do CDI; sem ele, usa config.risk_free como CDI constante.
    """
    params = params or {}
    total = len(data) * len(strategies)
    rows, done = [], 0
    for ticker, df in data.items():
        if should_stop and should_stop():
            break
        df = df.dropna(subset=["open", "close"])
        ppy = periods_per_year(df.index)
        years = (len(df) - 1) / ppy if len(df) > 1 else 0.0
        cdi_cagr = _cdi_cagr(cdi, df.index, ppy, config.risk_free)
        for strat in strategies:
            if should_stop and should_stop():
                break
            row = {"ticker": ticker, "strategy": strat.name, "years": years, "cdi_cagr": cdi_cagr, "error": ""}
            try:
                if len(df) < min_bars:
                    raise ValueError(f"histórico curto ({len(df)} pregões)")
                p = params.get(strat.name, strat.defaults())
                res = run_backtest(df, strat.generate_signals(df, **p), config, p)
                m, b = res.metrics, res.benchmark_metrics
                row.update(
                    sharpe=m["sharpe"], cagr=m["cagr"], bh_cagr=b["cagr"], total_return=m["total_return"],
                    max_drawdown=m["max_drawdown"], n_trades=m["n_trades"], exposure=m["exposure"],
                    trades_per_year=m["n_trades"] / years if years else np.nan,
                    excess_cdi=m["cagr"] - cdi_cagr, excess_bh=m["cagr"] - b["cagr"],
                )
            except Exception as e:  # um par com problema não derruba a varredura
                row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            done += 1
            if progress:
                progress(done, total)
    return pd.DataFrame(rows).reindex(columns=RESULT_COLUMNS)


def _cdi_cagr(cdi: pd.Series | None, dates: pd.DatetimeIndex, ppy: float, fallback: float) -> float:
    if cdi is not None and len(cdi) and len(dates) > 1:
        s = cdi.reindex(cdi.index.union(dates)).ffill().reindex(dates)
        if s.notna().iloc[0] and s.notna().iloc[-1]:
            years = (len(dates) - 1) / ppy
            return float((s.iloc[-1] / s.iloc[0]) ** (1 / max(years, 1e-9)) - 1)
    return float(fallback)


WIN_MARGIN = 0.001  # 0,1 ponto por ano: diferença menor que isso é empate, não vitória
BUY_AND_HOLD = "Buy & Hold"


def _win_rate(s: pd.Series) -> float:
    s = s.dropna()
    return float((s > WIN_MARGIN).mean()) if len(s) else float("nan")


def summarize_strategies(res: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por estratégia, ordenada pela robustez (% de ativos em que ganha do CDI)."""
    ok = res[res["error"].fillna("") == ""]
    if ok.empty:
        return pd.DataFrame()
    g = ok.groupby("strategy")
    out = pd.DataFrame({
        "ativos": g.size(),
        "vence_cdi": g["excess_cdi"].apply(_win_rate),
        "vence_bh": g["excess_bh"].apply(_win_rate),
        "excesso_cdi_mediano": g["excess_cdi"].median(),
        "sharpe_mediano": g["sharpe"].median(),
        "cagr_mediano": g["cagr"].median(),
        "dd_mediano": g["max_drawdown"].median(),
        "pior_dd": g["max_drawdown"].min(),
        "trades_ano": g["trades_per_year"].median(),
        "exposicao": g["exposure"].median(),
    })
    if BUY_AND_HOLD in out.index:
        out.loc[BUY_AND_HOLD, "vence_bh"] = np.nan  # comparar o Buy & Hold com ele mesmo não informa nada
    return out.sort_values(["vence_cdi", "excesso_cdi_mediano"], ascending=False).reset_index()


def summarize_assets(res: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por ativo: melhor estratégia, quantas vencem o CDI, Buy & Hold e CDI do período."""
    ok = res[res["error"].fillna("") == ""]
    if ok.empty:
        return pd.DataFrame()
    best = ok.loc[ok.groupby("ticker")["excess_cdi"].idxmax(), ["ticker", "strategy", "excess_cdi", "sharpe"]]
    best = best.rename(columns={"strategy": "melhor", "excess_cdi": "melhor_excesso", "sharpe": "melhor_sharpe"})
    g = ok.groupby("ticker")
    agg = pd.DataFrame({
        "estrategias": g.size(),
        "vencem_cdi": g["excess_cdi"].apply(_win_rate),
        "bh_cagr": g["bh_cagr"].first(),
        "cdi_cagr": g["cdi_cagr"].first(),
        "anos": g["years"].first(),
    }).reset_index()
    return agg.merge(best, on="ticker").sort_values(["vencem_cdi", "melhor_excesso"], ascending=False,
                                                    ignore_index=True)


def pivot(res: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Tabela ativo × estratégia de uma métrica (para o mapa de calor)."""
    ok = res[res["error"].fillna("") == ""]
    return ok.pivot_table(index="ticker", columns="strategy", values=metric, aggfunc="first")
