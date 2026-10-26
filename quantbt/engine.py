"""Motor de backtest.

Modelo de execução:
- A estratégia gera um sinal de posição alvo (fração do patrimônio, entre -1 e 1)
  usando dados até o fechamento do candle t.
- A ordem é executada na ABERTURA do candle t+1 (sem look-ahead).
- Só há negociação quando o alvo muda (não há rebalanceamento diário).
- Custos: comissão percentual + fixa por ordem e slippage percentual no preço.
- Caixa: opcionalmente rende o CDI (como num fundo brasileiro); saldo negativo paga a mesma taxa.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantbt.metrics import compute_metrics, periods_per_year


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission_pct: float = 0.0005  # fração do valor negociado (0.0005 = 0,05%)
    commission_fixed: float = 0.0  # valor fixo por ordem
    slippage_pct: float = 0.0005  # fração do preço
    allow_short: bool = True
    risk_free: float = 0.0  # taxa livre de risco anual para Sharpe/Sortino (ex.: 0.10 = CDI de 10% a.a.)
    cash_index: pd.Series | None = None  # índice acumulado do CDI: se dado, o caixa rende CDI dia a dia


@dataclass
class BacktestResult:
    equity: pd.Series
    benchmark: pd.Series
    signals: pd.Series
    exposure: pd.Series
    trades: pd.DataFrame
    fills: pd.DataFrame
    metrics: dict
    benchmark_metrics: dict
    config: BacktestConfig
    params: dict = field(default_factory=dict)


TRADE_COLUMNS = [
    "direction",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "quantity",
    "pnl",
    "return_pct",
    "bars",
    "status",
]


def _sign(x: float) -> int:
    return 1 if x > 1e-12 else (-1 if x < -1e-12 else 0)


class _TradeBook:
    """Agrupa execuções em operações (período com posição de mesmo sinal)."""

    def __init__(self) -> None:
        self.trades: list[dict] = []
        self.cur: dict | None = None

    def fill(self, i: int, t, qty: float, px: float, fee: float, pos_after: float) -> None:
        if abs(qty) < 1e-12:
            return
        if self.cur is None:
            self.cur = {
                "direction": _sign(qty),
                "entry_i": i,
                "entry_time": t,
                "entry_qty": 0.0,
                "entry_val": 0.0,
                "exit_qty": 0.0,
                "exit_val": 0.0,
                "cash": 0.0,
                "max_qty": 0.0,
            }
        c = self.cur
        c["cash"] += -qty * px - fee
        if _sign(qty) == c["direction"]:
            c["entry_qty"] += abs(qty)
            c["entry_val"] += abs(qty) * px
        else:
            c["exit_qty"] += abs(qty)
            c["exit_val"] += abs(qty) * px
        c["max_qty"] = max(c["max_qty"], abs(pos_after))
        if abs(pos_after) < 1e-12:
            self._close(i, t, status="fechada")

    def _close(self, i: int, t, status: str, mark_value: float = 0.0) -> None:
        c = self.cur
        assert c is not None
        pnl = c["cash"] + mark_value
        entry_price = c["entry_val"] / c["entry_qty"] if c["entry_qty"] else np.nan
        exit_price = c["exit_val"] / c["exit_qty"] if c["exit_qty"] else np.nan
        self.trades.append(
            {
                "direction": "Compra" if c["direction"] > 0 else "Venda",
                "entry_time": c["entry_time"],
                "entry_price": entry_price,
                "exit_time": t,
                "exit_price": exit_price,
                "quantity": c["max_qty"],
                "pnl": pnl,
                "return_pct": pnl / c["entry_val"] if c["entry_val"] else 0.0,
                "bars": i - c["entry_i"],
                "status": status,
            }
        )
        self.cur = None

    def finish(self, i: int, t, shares: float, last_close: float) -> None:
        """Marca a mercado uma operação ainda aberta no fim dos dados."""
        if self.cur is not None:
            c = self.cur
            c["exit_qty"] += abs(shares)
            c["exit_val"] += abs(shares) * last_close
            self._close(i, t, status="aberta", mark_value=shares * last_close)


def cash_returns(index_series: pd.Series | None, dates: pd.DatetimeIndex) -> np.ndarray:
    """Rendimento do caixa de cada candle em relação ao anterior, a partir de um índice acumulado (CDI)."""
    if index_series is None or len(index_series) == 0:
        return np.zeros(len(dates))
    s = index_series.sort_index()
    s = s[~s.index.duplicated(keep="last")]
    s = s.reindex(s.index.union(dates)).ffill().reindex(dates)
    g = (s / s.shift(1) - 1.0).fillna(0.0).to_numpy()
    return np.where(np.isfinite(g), g, 0.0)


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    config: BacktestConfig | None = None,
    params: dict | None = None,
) -> BacktestResult:
    config = config or BacktestConfig()
    if len(df) < 2:
        raise ValueError("São necessários pelo menos 2 candles para o backtest.")

    lo = -1.0 if config.allow_short else 0.0
    sig = (
        pd.Series(signals, index=df.index if not isinstance(signals, pd.Series) else None)
        .reindex(df.index)
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .fillna(0.0)
        .clip(lo, 1.0)
    )
    # alvo decidido no fechamento de t-1, executado na abertura de t
    target = sig.shift(1).fillna(0.0).to_numpy()

    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    idx = df.index
    n = len(df)
    cash_growth = cash_returns(config.cash_index, idx)  # rendimento do caixa entre um candle e o seguinte

    cash = float(config.initial_capital)
    shares = 0.0
    prev_target = 0.0
    equity = np.empty(n)
    exposure = np.zeros(n)
    book = _TradeBook()
    fills: list[dict] = []

    for i in range(n):
        if i:
            cash *= 1.0 + cash_growth[i]
        tgt = target[i]
        if tgt != prev_target and o[i] > 0 and np.isfinite(o[i]):
            eq_open = cash + shares * o[i]
            desired = tgt * max(eq_open, 0.0) / o[i]
            delta = desired - shares
            if abs(delta) > 1e-12:
                px = o[i] * (1.0 + config.slippage_pct * _sign(delta))
                notional = abs(delta) * px
                fee = notional * config.commission_pct + config.commission_fixed
                cash -= delta * px + fee
                # divide a execução em "fechar" e "abrir" quando há virada de mão
                if _sign(shares) != 0 and _sign(desired) != _sign(shares):
                    close_qty, open_qty = -shares, desired
                else:
                    close_qty, open_qty = 0.0, delta
                total = abs(close_qty) + abs(open_qty)
                if abs(close_qty) > 1e-12:
                    book.fill(i, idx[i], close_qty, px, fee * abs(close_qty) / total, 0.0)
                if abs(open_qty) > 1e-12:
                    book.fill(i, idx[i], open_qty, px, fee * abs(open_qty) / total, desired)
                shares = desired
                fills.append(
                    {"time": idx[i], "quantity": delta, "price": px, "fee": fee, "position": shares}
                )
            prev_target = tgt
        equity[i] = cash + shares * c[i]
        exposure[i] = (shares * c[i] / equity[i]) if equity[i] > 0 else 0.0

    book.finish(n - 1, idx[-1], shares, c[-1])

    equity_s = pd.Series(equity, index=idx, name="Estratégia")
    benchmark = pd.Series(
        config.initial_capital * c / c[0], index=idx, name="Buy & Hold"
    )
    trades = pd.DataFrame(book.trades, columns=TRADE_COLUMNS)
    fills_df = pd.DataFrame(fills, columns=["time", "quantity", "price", "fee", "position"])
    exposure_s = pd.Series(exposure, index=idx, name="Exposição")

    ppy = periods_per_year(idx)
    metrics = compute_metrics(equity_s, ppy, trades=trades, exposure=exposure_s, risk_free=config.risk_free)
    bench_metrics = compute_metrics(benchmark, ppy, risk_free=config.risk_free)

    return BacktestResult(
        equity=equity_s,
        benchmark=benchmark,
        signals=sig,
        exposure=exposure_s,
        trades=trades,
        fills=fills_df,
        metrics=metrics,
        benchmark_metrics=bench_metrics,
        config=config,
        params=dict(params or {}),
    )
