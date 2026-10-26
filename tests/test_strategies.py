import numpy as np
import pandas as pd
import pytest

from quantbt.data import sample_data
from quantbt.strategies import (
    BUILTIN, TurnOfMonth, adx, macd, parabolic_sar, stochastic, supertrend,
)


@pytest.mark.parametrize("strat", BUILTIN, ids=lambda s: s.name)
def test_no_lookahead(strat):
    """Mudar os preços DEPOIS do dia k não pode mudar nenhum sinal ATÉ o dia k."""
    df = sample_data(900, seed=11)
    k = 600
    alt = df.copy()
    rng = np.random.default_rng(0)
    noise = np.exp(rng.normal(0, 0.05, (len(df) - k - 1, 1)))
    alt.iloc[k + 1:, :4] = alt.iloc[k + 1:, :4].to_numpy() * noise
    a = strat.generate_signals(df, **strat.defaults())
    b = strat.generate_signals(alt, **strat.defaults())
    pd.testing.assert_series_equal(a.iloc[:k + 1], b.iloc[:k + 1], check_names=False)


@pytest.mark.parametrize("strat", BUILTIN, ids=lambda s: s.name)
def test_signals_valid_and_params_in_range(strat):
    df = sample_data(900, seed=12)
    sig = strat.generate_signals(df, **strat.defaults())
    assert len(sig) == len(df) and sig.notna().all()
    assert sig.between(-1, 1).all()  # posição: fração do patrimônio (alvo de volatilidade usa frações)
    for p in strat.params:
        assert p.min <= p.default <= p.max, p.name


def _trend(n=300, slope=0.004, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(slope + rng.normal(0, 0.004, n)))
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1.0}, index=idx)


def test_trend_indicators_follow_direction():
    up, down = _trend(), _trend(slope=-0.004)
    assert supertrend(up).iloc[-50:].eq(1).all() and supertrend(down).iloc[-50:].eq(-1).all()
    # o SAR vira de lado em recuos curtos: mede a média sobre várias séries (e a simetria alta/queda)
    sar_up = np.mean([parabolic_sar(_trend(seed=k)).iloc[50:].mean() for k in range(20)])
    sar_dn = np.mean([parabolic_sar(_trend(slope=-0.004, seed=k)).iloc[50:].mean() for k in range(20)])
    assert sar_up > 0.85 and sar_dn < -0.85 and abs(sar_up + sar_dn) < 0.05
    a, pdi, mdi = adx(up)
    assert (pdi.iloc[-50:] > mdi.iloc[-50:]).all() and a.dropna().between(0, 100).all()
    line, sig = macd(up["close"])
    assert (line.iloc[-50:] > 0).all()


def test_stochastic_bounds():
    k, d = stochastic(sample_data(500))
    assert k.dropna().between(0, 100).all() and d.dropna().between(0, 100).all()


def test_turn_of_month_holds_around_month_end():
    df = sample_data(300)
    sig = TurnOfMonth().generate_signals(df, before=1, after=3)
    idx = df.index
    # posição no candle t+1 = sinal em t; os candles mantidos devem ser os 3 primeiros e o último de cada mês
    held = sig.shift(1).fillna(0).astype(bool)
    month = pd.Series(idx.to_period("M"), index=idx)
    first3 = month.groupby(month).cumcount() < 3
    last1 = month.groupby(month).cumcount(ascending=False) < 1
    assert (held.iloc[1:] == (first3 | last1).iloc[1:]).all()
