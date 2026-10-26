import numpy as np
import pandas as pd
import pytest

from quantbt import external
from quantbt.correlation import analyze


def _fake_series(seed, start="2024-01-01", n=60, freq="D"):
    idx = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 0.0}, index=idx)


@pytest.fixture
def fake_load(monkeypatch):
    data = {"BTC": _fake_series(1), "DOLAR": _fake_series(2, freq="B"), "OURO": _fake_series(3, freq="B")}

    def load(code, start=None, end=None, refresh=False):
        df = data[code.upper()]
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        return df

    monkeypatch.setattr(external, "load", load)
    return data


def test_catalog_consistency():
    assert len(external.CATALOG) == len({a.alias for a in external._ASSETS})
    for preset in external.PRESETS.values():
        assert all(c in external.CATALOG for c in preset)
    for code in ("WDO", "DOL", "WIN", "IND", "BTC", "OURO", "PETROLEO", "DOLAR", "PTAX"):
        assert external.is_external(code) and external.describe(code)
    assert not external.is_external("PETR4")
    # substitutos dos futuros da B3 apontam para a série à vista correta
    assert external.CATALOG["WDO"].symbol == external.CATALOG["DOLAR"].symbol == "BRL=X"
    assert external.CATALOG["WIN"].symbol == external.CATALOG["IBOV"].symbol == "^BVSP"


def test_split_codes():
    b3c, ext = external.split_codes(["petr4", "BTC", "VALE3", "wdo"])
    assert b3c == ["PETR4", "VALE3"] and ext == ["BTC", "WDO"]


def test_align_to_b3_calendar_uses_last_price():
    ext = pd.DataFrame({"BTC": [1.0, 2.0, 3.0, 4.0]}, index=pd.to_datetime(
        ["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]))  # sex, sáb, dom, seg
    b3_days = pd.to_datetime(["2024-01-05", "2024-01-08"])
    out = external.align_to(b3_days, ext)
    assert out["BTC"].tolist() == [1.0, 4.0]  # fim de semana absorvido no retorno de segunda
    gap = external.align_to(pd.to_datetime(["2024-01-20"]), ext, max_gap=3)
    assert gap["BTC"].isna().all()  # dado velho demais não é usado


def test_combined_closes_mixes_b3_and_external(fake_load):
    b3_idx = pd.bdate_range("2024-01-02", periods=30)

    def b3_loader(ts):
        return pd.DataFrame({t: np.linspace(10, 20, len(b3_idx)) for t in ts}, index=b3_idx)

    panel = external.combined_closes(["PETR4", "BTC", "OURO"], b3_loader)
    assert panel.columns.tolist() == ["PETR4", "BTC", "OURO"]
    assert panel.index.equals(b3_idx)  # tudo no calendário da B3
    assert panel["BTC"].notna().all()


def test_load_closes_in_brl(fake_load, monkeypatch):
    usd = external.load_closes(["OURO"])["OURO"]
    brl = external.load_closes(["OURO"], in_brl=True)["OURO"]
    dol = fake_load["DOLAR"]["close"].reindex(usd.index).ffill()
    assert np.allclose(brl.dropna(), (usd * dol).dropna())


def test_analyze_with_external_columns(fake_load):
    b3_idx = pd.bdate_range("2024-01-02", periods=40)
    rng = np.random.default_rng(9)

    def b3_loader(ts):
        return pd.DataFrame({t: 50 * np.exp(np.cumsum(rng.normal(0, 0.01, len(b3_idx)))) for t in ts}, index=b3_idx)

    panel = external.combined_closes(["PETR4", "VALE3", "BTC", "DOLAR"], b3_loader)
    res = analyze(panel, min_obs=10, rolling_window=0)
    assert len(res.pairs) == 6


def test_same_underlying_marked_as_same_issuer():
    from quantbt.correlation import same_issuer
    assert same_issuer("DOLAR", "WDO") and same_issuer("WIN", "IBOV")
    assert not same_issuer("DOLAR", "EURO") and not same_issuer("BTC", "PETR4")
    assert same_issuer("PETR3", "PETR4")


def test_periods_per_year_crypto_vs_b3():
    from quantbt.metrics import periods_per_year
    crypto = pd.date_range("2020-01-01", "2023-01-01", freq="D")
    b3 = pd.bdate_range("2020-01-01", "2023-01-01")
    assert periods_per_year(crypto) == pytest.approx(365, abs=1)
    assert periods_per_year(b3) == pytest.approx(261, abs=2)
