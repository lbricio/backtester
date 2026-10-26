import numpy as np
import pandas as pd
import pytest

from quantbt.data import sample_data
from quantbt.engine import BacktestConfig
from quantbt.scan import pivot, run_scan, summarize_assets, summarize_strategies
from quantbt.strategies import BUILTIN, BuyAndHold, SmaCross

CFG = BacktestConfig(commission_pct=0, slippage_pct=0)


def _data():
    return {"AAA3": sample_data(800, seed=1), "BBB3": sample_data(800, seed=2), "CCC3": sample_data(40, seed=3)}


def _cdi(start="2015-01-01", periods=4000, daily=0.0004):
    idx = pd.bdate_range(start, periods=periods)
    return pd.Series(np.cumprod(np.full(len(idx), 1 + daily)), index=idx)


def test_scan_runs_every_pair_and_flags_short_history():
    res = run_scan(_data(), BUILTIN, CFG, cdi=_cdi())
    assert len(res) == 3 * len(BUILTIN)
    short = res[res.ticker == "CCC3"]
    assert short["error"].str.contains("histórico curto").all()
    ok = res[res.error == ""]
    assert ok["sharpe"].notna().all() and len(ok) == 2 * len(BUILTIN)


def test_buy_and_hold_matches_benchmark_and_cdi_reference():
    res = run_scan(_data(), [BuyAndHold()], CFG, cdi=_cdi(daily=0.0004))
    ok = res[res.error == ""]
    assert ok["excess_bh"].abs().max() < 0.02  # B&H da estratégia ≈ B&H de referência
    expected = (1.0004 ** 252.0) - 1  # CDI constante ~10,6% a.a. em base 252
    assert ok["cdi_cagr"].iloc[0] == pytest.approx(expected, rel=0.05)


def test_cdi_fallback_to_constant_rate():
    res = run_scan({"AAA3": sample_data(300)}, [SmaCross()], BacktestConfig(risk_free=0.12), cdi=None)
    assert res["cdi_cagr"].iloc[0] == pytest.approx(0.12)


def test_custom_params_and_stop():
    calls = []
    res = run_scan(_data(), [SmaCross()], CFG, params={SmaCross.name: {"fast": 5, "slow": 20, "short": 0}},
                   progress=lambda i, n: calls.append((i, n)), should_stop=lambda: len(calls) >= 1)
    assert len(res) == 1 and calls == [(1, 3)]


def test_summaries_and_pivot():
    res = run_scan(_data(), BUILTIN, CFG, cdi=_cdi())
    s = summarize_strategies(res)
    assert set(s["strategy"]) == {st.name for st in BUILTIN}
    assert s["vence_cdi"].between(0, 1).all() and (s["ativos"] == 2).all()
    a = summarize_assets(res)
    assert set(a["ticker"]) == {"AAA3", "BBB3"} and a["melhor"].notna().all()
    pv = pivot(res, "excess_cdi")
    assert pv.shape == (2, len(BUILTIN))


def test_ties_and_buy_and_hold_self_comparison():
    res = run_scan(_data(), [BuyAndHold(), SmaCross()], CFG, cdi=_cdi())
    s = summarize_strategies(res).set_index("strategy")
    assert pd.isna(s.loc["Buy & Hold", "vence_bh"])  # não se compara consigo mesmo
    tie = pd.DataFrame({"ticker": ["A", "B"], "strategy": ["X", "X"], "excess_cdi": [0.0004, 0.05],
                        "excess_bh": [0.0, 0.0], "sharpe": [1, 1], "cagr": [0.1, 0.1], "max_drawdown": [-0.1, -0.1],
                        "trades_per_year": [1, 1], "exposure": [1, 1], "bh_cagr": [0.1, 0.1],
                        "cdi_cagr": [0.1, 0.1], "years": [5, 5], "error": ["", ""]})
    assert summarize_strategies(tie).iloc[0]["vence_cdi"] == 0.5  # 0,04 p.p. é empate
