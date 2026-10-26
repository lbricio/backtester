import numpy as np
import pandas as pd
import pytest

from quantbt.data import normalize, sample_data
from quantbt.engine import BacktestConfig, run_backtest
from quantbt.optimizer import frange, grid_search
from quantbt.strategies import BUILTIN, CUSTOM_TEMPLATE, CodeStrategy, SmaCross

NO_COSTS = BacktestConfig(commission_pct=0, slippage_pct=0, initial_capital=1000)


def _df(prices):
    idx = pd.bdate_range("2024-01-01", periods=len(prices))
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 0.0}, index=idx)


def test_signal_executes_next_open_no_lookahead():
    df = _df([10, 10, 20, 20])
    df["open"] = [10, 10, 15, 20]
    sig = pd.Series([0, 1, 1, 1], index=df.index)  # decide no fechamento do candle 1
    res = run_backtest(df, sig, NO_COSTS)
    # compra a 15 na abertura do candle 2 -> 1000/15 ações, fecha a 20
    assert res.equity.iloc[1] == pytest.approx(1000)
    assert res.equity.iloc[-1] == pytest.approx(1000 / 15 * 20)


def test_buy_and_hold_matches_benchmark_without_costs():
    df = sample_data(300)
    sig = pd.Series(1.0, index=df.index)
    res = run_backtest(df, sig, NO_COSTS)
    # entra na abertura do 2º candle
    expected = 1000 / df["open"].iloc[1] * df["close"].iloc[-1]
    assert res.equity.iloc[-1] == pytest.approx(expected)
    assert len(res.trades) == 1 and res.trades.iloc[0]["status"] == "aberta"


def test_trade_pnl_sums_to_equity_change():
    df = sample_data(800)
    strat = SmaCross()
    cfg = BacktestConfig(initial_capital=10_000, commission_pct=0.001, slippage_pct=0.001)
    sig = strat.generate_signals(df, fast=10, slow=30, short=1)
    res = run_backtest(df, sig, cfg)
    assert res.trades["pnl"].sum() == pytest.approx(res.equity.iloc[-1] - 10_000, rel=1e-9)
    assert res.metrics["n_trades"] == len(res.trades) > 5


def test_short_and_flip():
    df = _df([10, 10, 8, 8, 12, 12])
    sig = pd.Series([0, -1, -1, 1, 1, 1], index=df.index)
    res = run_backtest(df, sig, NO_COSTS)
    t = res.trades
    assert list(t["direction"]) == ["Venda", "Compra"]
    # sinal do candle 1 -> vende 125 ações a 8 no candle 2; virada no candle 4 -> recompra a 12
    assert t.iloc[0]["pnl"] == pytest.approx(-125 * (12 - 8))
    assert t.iloc[1]["status"] == "aberta" and t.iloc[1]["entry_price"] == pytest.approx(12)
    no_short = run_backtest(df, sig, BacktestConfig(allow_short=False, commission_pct=0, slippage_pct=0))
    assert list(no_short.trades["direction"]) == ["Compra"]


def test_costs_reduce_equity():
    df = sample_data(500)
    sig = SmaCross().generate_signals(df, fast=5, slow=20)
    free = run_backtest(df, sig, NO_COSTS)
    costly = run_backtest(df, sig, BacktestConfig(initial_capital=1000, commission_pct=0.002, slippage_pct=0.002))
    assert costly.equity.iloc[-1] < free.equity.iloc[-1]


@pytest.mark.parametrize("strat", BUILTIN, ids=lambda s: s.name)
def test_builtin_strategies_run(strat):
    df = sample_data(600)
    res = run_backtest(df, strat.generate_signals(df, **strat.defaults()))
    assert np.isfinite(res.equity).all()


def test_custom_strategy_template():
    s = CodeStrategy(CUSTOM_TEMPLATE)
    assert [p.name for p in s.params] == ["trend", "atr_period", "max_atr_pct"]
    df = sample_data(600)
    res = run_backtest(df, s.generate_signals(df, **s.defaults()))
    assert len(res.equity) == 600


def test_custom_strategy_errors():
    with pytest.raises(ValueError):
        CodeStrategy("x = 1")


def test_grid_search():
    df = sample_data(400)
    out = grid_search(df, SmaCross(), {"fast": [5, 10], "slow": [20, 40]}, NO_COSTS, {"short": 0})
    assert len(out) == 4 and "sharpe" in out


def test_frange():
    assert frange(1, 5, 2, True) == [1, 3, 5]
    assert frange(0.1, 0.3, 0.1, False) == pytest.approx([0.1, 0.2, 0.3])


def test_normalize_aliases():
    raw = pd.DataFrame({"Data": ["2024-01-02", "2024-01-01"], "Fechamento": [2, 1], "Volume": [5, 6]})
    df = normalize(raw)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.is_monotonic_increasing and df["close"].tolist() == [1, 2]


def test_cash_earns_cdi_when_out_of_market():
    df = _df([10] * 253)
    idx = df.index
    cdi = pd.Series((1.0004) ** np.arange(len(idx)), index=idx)  # ~10,6% a.a.
    flat = pd.Series(0.0, index=idx)
    res = run_backtest(df, flat, BacktestConfig(initial_capital=1000, cash_index=cdi, commission_pct=0, slippage_pct=0))
    assert res.equity.iloc[-1] == pytest.approx(1000 * 1.0004 ** 252)
    no_cdi = run_backtest(df, flat, NO_COSTS)
    assert no_cdi.equity.iloc[-1] == pytest.approx(1000)


def test_cash_index_with_gaps_compounds():
    df = _df([10, 10, 10])
    df.index = pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04"])  # pregão pulou o dia 2
    cdi = pd.Series([1.0, 1.01, 1.02, 1.03], index=pd.to_datetime(
        ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]))  # o CDI rendeu no dia 2 também
    res = run_backtest(df, pd.Series(0.0, index=df.index), BacktestConfig(initial_capital=100, cash_index=cdi))
    assert res.equity.iloc[-1] == pytest.approx(103)
