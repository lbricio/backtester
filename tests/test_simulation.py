import math

import numpy as np
import pandas as pd
import pytest

from quantbt import montecarlo as mc
from quantbt import options as op
from quantbt import valuation as va


def _prices(n=600, vol=0.02, drift=0.0005, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(50 * np.exp(np.cumsum(rng.normal(drift, vol, n))), index=idx)


# ---------------------------------------------------------------- Monte Carlo de preço
def test_estimate_annualizes():
    est = mc.estimate(_prices(), years=None)
    assert est.sigma_annual == pytest.approx(0.02 * math.sqrt(est.ppy), rel=0.1)
    assert est.s0 == pytest.approx(_prices().iloc[-1])


@pytest.mark.parametrize("model", ["gbm", "t", "bootstrap", "block"])
def test_zero_drift_keeps_expected_price(model):
    est = mc.estimate(_prices(), years=None)
    cfg = mc.MCConfig(model=model, horizon=60, n_paths=40_000, drift="zero", seed=1)
    paths = mc.simulate(est, cfg)
    assert paths.shape == (40_000, 61) and (paths[:, 0] == est.s0).all()
    assert paths[:, -1].mean() == pytest.approx(est.s0, rel=0.02)
    # volatilidade diária simulada ≈ histórica
    sim_vol = np.log(paths[:, 1:] / paths[:, :-1]).std()
    assert sim_vol == pytest.approx(est.sigma_d, rel=0.05)


def test_risk_free_drift_grows_at_rate():
    est = mc.estimate(_prices(), years=None)
    cfg = mc.MCConfig(horizon=252, n_paths=40_000, drift="risk_free", risk_free=0.12, seed=2)
    paths = mc.simulate(est, cfg)
    expected = est.s0 * (1.12) ** (252 / est.ppy)
    assert paths[:, -1].mean() == pytest.approx(expected, rel=0.03)


def test_vol_override_and_fat_tails():
    est = mc.estimate(_prices(), years=None)
    p = mc.simulate(est, mc.MCConfig(horizon=20, n_paths=20_000, drift="zero", vol_override=0.60, seed=3))
    assert np.log(p[:, 1:] / p[:, :-1]).std() * math.sqrt(est.ppy) == pytest.approx(0.60, rel=0.05)
    pt = mc.simulate(est, mc.MCConfig(model="t", horizon=20, n_paths=20_000, drift="zero", seed=3))
    pn = mc.simulate(est, mc.MCConfig(model="gbm", horizon=20, n_paths=20_000, drift="zero", seed=3))
    kurt = lambda x: ((x - x.mean()) ** 4).mean() / x.var() ** 2  # noqa: E731
    rt, rn = np.log(pt[:, 1:] / pt[:, :-1]).ravel(), np.log(pn[:, 1:] / pn[:, :-1]).ravel()
    assert kurt(rt) > kurt(rn) + 1  # t-Student tem caudas mais gordas


def test_batches_sum_to_total_and_stats():
    est = mc.estimate(_prices(), years=None)
    cfg = mc.MCConfig(horizon=30, n_paths=1000, seed=4)
    parts = list(mc.simulate_batches(est, cfg, batches=7))
    assert sum(len(p) for p in parts) == 1000
    paths = np.vstack(parts)
    st = mc.stats(paths, est.s0, target=est.s0 * 1.1)
    assert st.n == 1000 and 0 <= st.prob_up <= 1
    assert st.prob_touch_target >= st.prob_above_target  # tocar no caminho ≥ terminar acima
    assert st.cvar95 <= st.var95 <= 0 or st.var95 > 0  # CVaR é pior ou igual ao VaR
    assert st.pcts[5][-1] <= st.pcts[50][-1] <= st.pcts[95][-1]


def test_short_history_raises():
    with pytest.raises(ValueError):
        mc.estimate(_prices(n=10), years=None)


# ---------------------------------------------------------------- opções
def _opt(kind="call", **kw):
    base = dict(s=100.0, k=100.0, days=126, r=0.10, sigma=0.30, q=0.0)
    base.update(kw)
    return op.OptionSpec(kind, **base)


def test_black_scholes_known_value_and_parity():
    c, p = op.black_scholes(_opt("call")), op.black_scholes(_opt("put"))
    o = _opt()
    # paridade put-call: C - P = S e^{-qT} - K e^{-rT}
    assert c - p == pytest.approx(o.s - o.k * math.exp(-o.rc * o.t), abs=1e-9)
    # valor de referência (S=K=100, T=0,5, r_c=ln 1,1, σ=30%)
    assert c == pytest.approx(10.7838, abs=1e-3)  # conferido com cálculo independente


def test_monte_carlo_matches_black_scholes():
    for kind in ("call", "put"):
        o = _opt(kind, k=110)
        res = op.mc_price(o, 400_000, seed=5)
        assert abs(res.price - op.black_scholes(o)) < 3 * res.stderr + 1e-3


def test_pricer_convergence_history_grows():
    pr = op.MCOptionPricer(_opt(), seed=6)
    for _ in range(5):
        res = pr.step(10_000)
    assert [h[0] for h in res.history] == [20_000, 40_000, 60_000, 80_000, 100_000]
    assert res.history[-1][2] < res.history[0][2]  # erro cai com mais simulações


def test_greeks_match_finite_differences():
    o = _opt()
    g = op.greeks(o)
    h = 0.01
    up, dn = op.black_scholes(_opt(s=100 + h)), op.black_scholes(_opt(s=100 - h))
    assert g["delta"] == pytest.approx((up - dn) / (2 * h), rel=1e-4)
    vup = op.black_scholes(_opt(sigma=0.31))
    assert g["vega"] == pytest.approx(vup - op.black_scholes(o), rel=0.02)  # por 1 p.p.
    assert 0 < g["prob_exercicio"] < 1


def test_implied_vol_roundtrip():
    o = _opt("put", k=95, sigma=0.42)
    price = op.black_scholes(o)
    assert op.implied_vol(_opt("put", k=95, sigma=0.2), price) == pytest.approx(0.42, abs=1e-5)
    assert op.implied_vol(o, 1e6) is None  # preço impossível


def test_expired_option_is_intrinsic():
    assert op.black_scholes(_opt("call", s=120, days=0)) == pytest.approx(20)
    assert op.black_scholes(_opt("put", s=120, days=0)) == 0


# ---------------------------------------------------------------- valuation
def test_formulas():
    assert va.graham(4.0, 25.0) == pytest.approx(math.sqrt(22.5 * 100))
    assert va.graham(-1.0, 25.0) is None
    assert va.bazin(3.0, 0.06) == pytest.approx(50.0)
    assert va.gordon(2.0, 0.12, 0.04) == pytest.approx(2.0 * 1.04 / 0.08)
    assert va.gordon(2.0, 0.04, 0.05) is None  # k <= g
    # FCD sem crescimento e perpetuidade 0 = perpetuidade simples FCF/k
    assert va.dcf(5.0, 0.0, 5, 0.0, 0.10) == pytest.approx(50.0)
    assert va.margin(120, 100) == pytest.approx(0.2)


def test_dcf_monte_carlo_centers_on_point_estimate():
    base = va.dcf(5.0, 0.08, 5, 0.03, 0.12)
    dist = va.dcf_monte_carlo(5.0, 0.08, 5, 0.03, 0.12, 0.01, 0.005, 0.005, n=50_000, seed=7)
    assert len(dist) > 45_000
    assert np.median(dist) == pytest.approx(base, rel=0.05)
    assert np.percentile(dist, 5) < base < np.percentile(dist, 95)


def test_dividends_per_share():
    idx = pd.to_datetime(["2020-05-01", "2021-05-01", "2022-05-01", "2023-05-01", "2024-05-01",
                          "2025-05-01", "2025-11-01"])
    divs = pd.Series([1, 1, 1, 1, 1, 1, 1], index=idx, dtype=float)
    last12, avg5 = va.dividends_per_share(divs, today=pd.Timestamp("2026-01-15"))
    assert last12 == 2.0  # mai/2025 e nov/2025 estão nos 12 meses anteriores a 15/01/2026
    assert avg5 == pytest.approx((1 + 1 + 1 + 1 + 2) / 5)  # 2021–2025


def test_evaluate_table():
    f = va.Fundamentals(ticker="TEST3", price=40, eps=4, bvps=25, dps_12m=2.4, dps_avg5=2.0, fcf_ps=3.0)
    t = va.evaluate(f, va.ValuationParams())
    assert list(t["metodo"]) == ["Graham", "Bazin", "Gordon", "Fluxo de caixa descontado"]
    assert t.loc[t.metodo == "Bazin", "preco_justo"].item() == pytest.approx(2.0 / 0.06)
    assert t["margem"].notna().all()
