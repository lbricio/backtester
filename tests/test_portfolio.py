import numpy as np
import pandas as pd
import pytest

from quantbt.data import sample_data
from quantbt.engine import BacktestConfig
from quantbt.portfolio import (
    PORTFOLIO_STRATEGIES, EqualWeight, PairsTrading, XSMomentum, liquidity_mask, make_panels, rebalance_dates,
    run_portfolio,
)

FREE = BacktestConfig(initial_capital=1000, commission_pct=0, slippage_pct=0)


def _universe(n_assets=12, n=900):
    return {f"A{i:02d}3": sample_data(n, seed=i) for i in range(n_assets)}


def test_equal_weight_two_constant_assets_flat_and_costs():
    idx = pd.bdate_range("2020-01-01", periods=400)
    flat = pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1e6}, index=idx)
    p = make_panels({"X": flat, "Y": flat.copy()})
    res = run_portfolio(p, EqualWeight(), FREE)
    assert res.equity.iloc[-1] == pytest.approx(1000)
    w = res.weights.iloc[-1]
    assert w["X"] == pytest.approx(0.5) and w["Y"] == pytest.approx(0.5)
    costly = run_portfolio(p, EqualWeight(), BacktestConfig(initial_capital=1000, commission_pct=0.01, slippage_pct=0))
    assert costly.equity.iloc[-1] < 1000 and costly.costs > 0


def test_cash_earns_cdi_when_strategy_holds_nothing():
    data = _universe(4, 400)
    p = make_panels(data)
    idx = p.close.index
    cdi = pd.Series(1.0004 ** np.arange(len(idx)), index=idx)

    class Nothing(EqualWeight):
        def target_weights(self, close, dates, eligible):
            return pd.DataFrame(0.0, index=dates, columns=close.columns)

    res = run_portfolio(p, Nothing(), BacktestConfig(initial_capital=1000, cash_index=cdi))
    days = len(res.equity) - 1
    assert res.equity.iloc[-1] == pytest.approx(1000 * 1.0004 ** days, rel=1e-6)


@pytest.mark.parametrize("strat", [s for s in PORTFOLIO_STRATEGIES if not s.long_short], ids=lambda s: s.name)
def test_no_lookahead_portfolio(strat):
    data = _universe(10, 800)
    p = make_panels(data)
    k = 600
    alt = {t: d.copy() for t, d in data.items()}
    rng = np.random.default_rng(1)
    for d in alt.values():
        d.iloc[k + 1:, :4] = d.iloc[k + 1:, :4].to_numpy() * np.exp(rng.normal(0, 0.05, (len(d) - k - 1, 1)))
    pa = make_panels(alt)
    dates = rebalance_dates(p.close.index, "M")
    dates = dates[dates <= p.close.index[k]]
    elig = p.close.notna()
    a = strat.target_weights(p.close, dates, elig, **strat.defaults())
    b = strat.target_weights(pa.close, dates, elig, **strat.defaults())
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("strat", PORTFOLIO_STRATEGIES, ids=lambda s: s.name)
def test_all_strategies_run(strat):
    res = run_portfolio(make_panels(_universe(10, 800)), strat, FREE)
    assert np.isfinite(res.equity).all() and res.metrics["rebalances"] >= 1
    # no rebalanceamento a exposição bruta é ≤ 100%; entre ajustes, o long-short pode derivar um pouco
    assert (res.weights.abs().sum(axis=1) <= (1.5 if strat.long_short else 1.0 + 1e-6)).all()
    if not strat.long_short:
        assert (res.weights >= -1e-9).all().all()


def test_momentum_picks_winners():
    idx = pd.bdate_range("2020-01-01", periods=400)
    mk = lambda drift: pd.DataFrame({c: 100 * np.exp(np.arange(400) * drift) for c in ("open", "high", "low", "close")} | {"volume": 1e6}, index=idx)  # noqa: E731
    p = make_panels({"WIN3": mk(0.002), "MID3": mk(0.0), "LOS3": mk(-0.002)})
    w = XSMomentum().target_weights(p.close, rebalance_dates(idx, "M")[-1:], p.close.notna(), top=1)
    assert w.iloc[-1].idxmax() == "WIN3" and w.iloc[-1].max() == 1.0


def test_pairs_market_neutral_and_mean_reverting():
    rng = np.random.default_rng(3)
    n = 900
    idx = pd.bdate_range("2020-01-01", periods=n)
    common = np.cumsum(rng.normal(0, 0.01, n))
    spread = np.zeros(n)
    for i in range(1, n):  # spread que volta à média (Ornstein-Uhlenbeck)
        spread[i] = 0.9 * spread[i - 1] + rng.normal(0, 0.01)
    a = 50 * np.exp(common + spread)
    b = 30 * np.exp(common)
    mk = lambda x: pd.DataFrame({"open": x, "high": x, "low": x, "close": x, "volume": 1e6}, index=idx)  # noqa: E731
    p = make_panels({"AAA3": mk(a), "BBB3": mk(b)})
    res = run_portfolio(p, PairsTrading(), FREE)
    w = res.weights
    held = w.abs().sum(axis=1) > 0
    assert held.any()
    assert ((np.sign(w.loc[held, "AAA3"]) * np.sign(w.loc[held, "BBB3"])) <= 0).all()  # um comprado, outro vendido
    assert res.equity.iloc[-1] > 1000  # spread que reverte gera lucro


def test_liquidity_mask_ranks_by_trailing_volume():
    idx = pd.bdate_range("2020-01-01", periods=120)
    d = lambda v: pd.DataFrame({"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": v}, index=idx)  # noqa: E731
    p = make_panels({"BIG3": d(1e8), "MID3": d(1e7), "SML3": d(1e5)})
    m = liquidity_mask(p, top=2)
    assert m.iloc[-1].to_dict() == {"BIG3": True, "MID3": True, "SML3": False}
