"""Testes das correções da revisão de código."""

import json
import time

import numpy as np
import pandas as pd
import pytest

from quantbt import b3, external
from quantbt.correlation import analyze
from quantbt.data import fix_yahoo_fx, sample_data
from quantbt.engine import BacktestConfig, run_backtest
from quantbt.metrics import compute_metrics
from quantbt.optimizer import MAX_COMBINATIONS, count_combinations, grid_search
from quantbt.strategies import SmaCross, rsi


# ---------------------------------------------------------------- otimizador
def test_grid_limit_checked_before_building_combinations():
    huge = {f"p{i}": list(range(100)) for i in range(6)}  # 10^12 combinações
    assert count_combinations(huge) == 100**6 > MAX_COMBINATIONS
    t = time.time()
    with pytest.raises(ValueError, match="limite"):
        grid_search(sample_data(50), SmaCross(), huge, BacktestConfig())
    assert time.time() - t < 1  # falha na hora, sem tentar gerar a lista


# ---------------------------------------------------------------- métricas
def test_risk_free_lowers_sharpe():
    idx = pd.bdate_range("2020-01-01", periods=504)
    rng = np.random.default_rng(0)
    eq = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, len(idx)))), index=idx)
    gross = compute_metrics(eq, 252)["sharpe"]
    net = compute_metrics(eq, 252, risk_free=0.10)["sharpe"]
    assert net < gross


def test_risk_free_flows_from_config():
    df = sample_data(400)
    sig = pd.Series(1.0, index=df.index)
    a = run_backtest(df, sig, BacktestConfig(risk_free=0.0)).metrics["sharpe"]
    b = run_backtest(df, sig, BacktestConfig(risk_free=0.12)).metrics["sharpe"]
    assert b < a


# ---------------------------------------------------------------- indicadores
def test_rsi_is_100_without_losses():
    s = pd.Series(np.arange(1.0, 40.0))  # só sobe
    r = rsi(s, 14)
    assert r.dropna().eq(100).all() and r.notna().sum() > 20


# ---------------------------------------------------------------- Yahoo
def test_fix_yahoo_fx_only_for_fx():
    idx = pd.bdate_range("2024-01-01", periods=4)
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    assert fix_yahoo_fx(s, "BRL=X").tolist() == [2.0, 3.0, 4.0]
    assert fix_yahoo_fx(s, "PETR4.SA").tolist() == [1.0, 2.0, 3.0, 4.0]


# ---------------------------------------------------------------- base B3
def _store_row(date, ticker, isin, spec="ON", close=10.0):
    return {"date": pd.Timestamp(date), "ticker": ticker, "name": "X", "spec": spec, "bdi": "02",
            "open": close, "high": close, "low": close, "close": close, "avg": close,
            "trades": 1, "quantity": 100, "volume": 1e6, "isin": isin}


def test_rights_filtered_and_reused_ticker_split(tmp_path, monkeypatch):
    monkeypatch.setattr(b3, "store_dir", lambda: tmp_path)
    b3._symbol_cache.clear()
    rows = [
        # FII antigo com o código FIXX11 (ISIN A), encerrado em 2019
        *[_store_row(f"2019-03-{d:02d}", "FIXX11", "BRFIXXCTF004", "CI", 100 + d) for d in range(1, 6)],
        # ETF novo que reaproveitou o código (ISIN B)
        *[_store_row(f"2019-12-{d:02d}", "FIXX11", "BRFIXXCTF012", "CI", 50 + d) for d in range(2, 7)],
        # direito de subscrição: deve sumir
        _store_row("2019-12-03", "ALZR12", "BRALZRD01M13", "DIR", 1.0),
    ]
    b3._write_parquet(pd.DataFrame(rows), b3.year_path(2019), index=False)
    assert not list(tmp_path.glob("*.tmp"))  # gravação atômica não deixa temporários

    store = b3.load_store()
    assert "ALZR12" not in set(store["ticker"])
    assert "spec" in b3.load_store(columns=["ticker", "spec"])
    assert "spec" not in b3.load_store(columns=["ticker"])

    new = b3.load_ticker("FIXX11", adjust=False)
    assert new.index.min() == pd.Timestamp("2019-12-02")  # só a história do ETF
    old = b3.load_ticker("FIXX11~2019", adjust=False)
    assert old["close"].iloc[0] == 101
    b3._symbol_cache.clear()


def test_build_symbol_map_renames_only_older_isin():
    df = pd.DataFrame({
        "date": pd.to_datetime(["1998-01-02", "2010-01-04", "2010-01-05", "2024-01-02"]),
        "ticker": ["PET 4", "PETR4", "ABCD3", "ABCD3"],
        "isin": ["BRPETRACNPR6", "BRPETRACNPR6", "BRABCDACNOR1", "BRABCDACNOR9"],
    })
    m = b3.build_symbol_map(df)
    assert m["BRPETRACNPR6"] == "PETR4"  # troca de código: mesma série
    assert m["BRABCDACNOR9"] == "ABCD3" and m["BRABCDACNOR1"] == "ABCD3~2010"


# ---------------------------------------------------------------- externos
@pytest.fixture
def ext_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(external, "cache_dir", lambda: tmp_path)
    calls = []

    def fake_fetch(asset, start):
        calls.append(start)
        idx = pd.bdate_range("2017-11-09", periods=30)  # ativo que "nasceu" depois do início pedido
        c = np.linspace(1, 2, len(idx))
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 0.0}, index=idx)

    monkeypatch.setattr(external, "_fetch", fake_fetch)
    return calls


def test_cache_remembers_requested_start(ext_cache):
    external.load("ETH", start="2016-10-25")
    external.load("ETH", start="2016-10-25")
    assert len(ext_cache) == 1  # antes rebaixava a cada chamada (ETH começa em 2017)
    meta = json.loads(next(external.cache_dir().glob("*.json")).read_text())
    assert meta["start"] == "2016-10-25"


def test_cache_fallback_when_offline(ext_cache, monkeypatch):
    external.load("ETH", start="2016-10-25")

    def offline(asset, start):
        raise OSError("sem internet")

    monkeypatch.setattr(external, "_fetch", offline)
    df = external.load("ETH", start="2016-10-25", refresh=True)  # usa o cache
    assert len(df) == 30
    with pytest.raises(OSError):
        external.load("ETH", start="2016-10-25", refresh=True, strict=True)


def test_usd_indices_convert_to_brl():
    assert external.CATALOG["SP500"].currency == "USD"
    assert external.CATALOG["VIX"].currency != "USD"  # nível, não preço


# ---------------------------------------------------------------- correlação
def test_same_issuer_vectorized_and_window_stored():
    rng = np.random.default_rng(5)
    idx = pd.bdate_range("2022-01-03", periods=300)
    base = rng.normal(0, 0.01, len(idx))
    cols = {"PETR3": base, "PETR4": base + rng.normal(0, 0.002, len(idx)),
            "DOLAR": rng.normal(0, 0.01, len(idx)), "WDO": rng.normal(0, 0.01, len(idx)),
            "VALE3": rng.normal(0, 0.01, len(idx))}
    prices = 100 * np.exp(pd.DataFrame(cols, index=idx).cumsum())
    res = analyze(prices, min_obs=30, rolling_window=40)
    flag = {frozenset((r.a, r.b)): r.same_issuer for r in res.pairs.itertuples()}
    assert flag[frozenset(("PETR3", "PETR4"))] and flag[frozenset(("DOLAR", "WDO"))]
    assert not flag[frozenset(("PETR3", "VALE3"))] and not flag[frozenset(("DOLAR", "VALE3"))]
    assert res.window == 40
