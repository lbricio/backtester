import numpy as np
import pandas as pd
import pytest

from quantbt import b3
from quantbt.correlation import analyze, cluster_order, to_returns


def _line(date, ticker, close, open_=None, bdi="02", tpmerc="010", fat=1, volume=1000.0, isin=None):
    isin = isin or f"BR{ticker[:4]:X<4}ACNOR{len(ticker)}"[:12].ljust(12, "0")
    open_ = close if open_ is None else open_
    p = lambda v: f"{int(round(v * 100)):013d}"
    s = (
        "01" + date + bdi + f"{ticker:<12}" + tpmerc + f"{'EMPRESA':<12}" + f"{'ON NM':<10}"
        + " " * 3 + " " * 4 + p(open_) + p(close * 1.01) + p(close * 0.99) + p(close) + p(close)
        + p(close) + p(close) + f"{10:05d}" + f"{100:018d}" + f"{int(volume * 100):018d}"
        + "0" * 13 + "0" + "99991231" + f"{fat:07d}" + "0" * 13 + f"{isin:<12}" + "001"
    )
    assert len(s) == 245, len(s)
    return s.encode()


def test_parse_cotahist_filters_and_scales():
    raw = b"\r\n".join([
        b"00COTAHIST.2024BOVESPA 20241231" + b" " * 214,
        _line("20240102", "PETR4", 38.5),
        _line("20240102", "PETR4F", 38.5, bdi="96"),  # fracionário: descartado
        _line("20240102", "PETRA40", 1.0, tpmerc="070"),  # opção: descartada
        _line("20240102", "XPTO3", 2500.0, fat=1000),  # cotado por lote de mil
        b"99COTAHIST.2024BOVESPA 20241231" + b" " * 214,
    ])
    df = b3.parse_cotahist(raw)
    assert sorted(df["ticker"].astype(str)) == ["PETR4", "XPTO3"]
    petr = df[df["ticker"] == "PETR4"].iloc[0]
    assert petr["close"] == pytest.approx(38.5)
    assert petr["date"] == pd.Timestamp("2024-01-02")
    assert df[df["ticker"] == "XPTO3"].iloc[0]["close"] == pytest.approx(2.5)


def test_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(b3, "store_dir", lambda: tmp_path)
    days = pd.bdate_range("2023-01-02", periods=30)
    lines = []
    for i, d in enumerate(days):
        px = 10 + i * 0.1
        if i >= 15:  # desdobramento 2:1 no 16º pregão
            px /= 2
        # código antigo até o 5º pregão, mesmo ISIN (como 'PET 4' -> PETR4)
        code = "ABC 3" if i < 5 else "ABCD3"
        lines.append(_line(d.strftime("%Y%m%d"), code, px, volume=1e6, isin="BRABCDACNOR1"))
        lines.append(_line(d.strftime("%Y%m%d"), "WXYZ3", 50 + i, volume=1e3))
    f = tmp_path / "COTAHIST_A2023.TXT"
    f.write_bytes(b"\r\n".join(lines))
    assert b3.import_file(f) == (2023, 60)
    assert b3.available_years() == [2023]

    raw = b3.load_ticker("ABCD3", adjust=False)
    assert len(raw) == 30  # inclui o período com o código antigo
    adj = b3.load_ticker("ABCD3")
    assert raw["close"].pct_change().min() < -0.4
    assert adj["close"].pct_change().min() > -0.05  # salto corrigido
    assert adj["close"].iloc[-1] == pytest.approx(raw["close"].iloc[-1])

    assert b3.liquid_universe(5).empty  # só 30 pregões: abaixo do histórico mínimo padrão (1 ano)
    uni = b3.liquid_universe(5, min_history=20)
    assert uni["ticker"].tolist() == ["ABCD3", "WXYZ3"]
    panel = b3.load_closes(["WXYZ3", "ABCD3", "NADA3"])
    assert panel.columns.tolist() == ["WXYZ3", "ABCD3"]


def test_adjust_splits_ignores_normal_moves():
    df = pd.DataFrame({"open": [10, 10.5, 7.9], "high": 0, "low": 0, "close": [10, 10.5, 7.2]})
    out = b3.adjust_splits(df)  # queda de 25%: movimento normal, não é evento
    assert out["close"].tolist() == df["close"].tolist()


def test_adjust_splits_bonus_50pct():
    # bonificação de 50% (3 por 2), como ITUB4 em 11/2018: 52,72 -> abre 34,99
    df = pd.DataFrame({"open": [52.5, 52.54, 34.99], "high": 0, "low": 0, "close": [52.6, 52.72, 34.69]})
    out = b3.adjust_splits(df)
    assert out["close"].iloc[1] == pytest.approx(52.72 * 2 / 3)


def test_to_reais():
    idx = pd.to_datetime(["1993-07-30", "1993-08-02", "1994-07-01", "1994-07-04"])
    df = pd.DataFrame({"date": idx, "close": [2_750_000.0, 2750.0, 2750.0, 1.0]})
    assert b3.to_reais(df)["close"].tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def _panel():
    rng = np.random.default_rng(0)
    n = 500
    base = rng.normal(0, 0.01, n)
    r = pd.DataFrame({
        "A": base + rng.normal(0, 0.002, n),
        "B": base + rng.normal(0, 0.002, n),  # quase igual a A
        "C": -base + rng.normal(0, 0.004, n),  # inverso
        "D": rng.normal(0, 0.01, n),  # independente
    }, index=pd.bdate_range("2020-01-01", periods=n))
    return 100 * np.exp(r.cumsum())


def test_analyze_ranks_pairs():
    res = analyze(_panel(), min_obs=30, rolling_window=60)
    top = res.pairs.iloc[0]
    assert {top["a"], top["b"]} == {"A", "B"} and top["corr"] > 0.9
    worst = res.pairs.iloc[-1]
    assert "C" in (worst["a"], worst["b"]) and worst["corr"] < -0.8
    assert len(res.pairs) == 6 and {"roll_mean", "roll_std"} <= set(res.pairs)


def test_analyze_weekly_and_coverage():
    p = _panel()
    p["E"] = np.nan
    p.loc[p.index[-10:], "E"] = 1.0
    res = analyze(p, freq="W-FRI", min_obs=20)
    assert res.dropped == ["E"]  # só 10 dias de dados: não forma nenhum par com 20 obs.
    assert len(res.returns) < 120


def test_cluster_order_groups_similar():
    res = analyze(_panel(), min_obs=30, rolling_window=0)
    order = cluster_order(res.matrix)
    assert abs(order.index("A") - order.index("B")) == 1


def test_returns_not_prices():
    p = _panel()
    r = to_returns(p)
    assert len(r) == len(p) - 1 and r.abs().max().max() < 0.2


def test_pairwise_corr_matches_pandas_with_gaps():
    from quantbt.correlation import pairwise_corr
    r = to_returns(_panel())
    r.iloc[10:60, 0] = np.nan
    r.iloc[200:230, 3] = np.nan
    fast, n = pairwise_corr(r, min_obs=10)
    slow = r.corr(min_periods=10)
    assert np.allclose(fast.to_numpy(), slow.to_numpy(), atol=1e-9)
    assert n.loc["A", "D"] == (r["A"].notna() & r["D"].notna()).sum()


def test_same_issuer_flag():
    p = _panel().rename(columns={"A": "PETR3", "B": "PETR4", "C": "VALE3", "D": "ITUB4"})
    res = analyze(p, min_obs=30, rolling_window=20, rolling_top=2)
    top = res.pairs.iloc[0]
    assert top["same_issuer"] and {top["a"], top["b"]} == {"PETR3", "PETR4"}
    assert res.pairs["roll_mean"].notna().sum() == 2  # só os 2 pares mais fortes


def _rets_df():
    return to_returns(_panel())


def test_sign_concordance_basic():
    from quantbt.correlation import sign_concordance
    idx = pd.bdate_range("2024-01-01", periods=6)
    r = pd.DataFrame({"X": [0.01, -0.02, 0.03, -0.01, 0.0005, 0.02],
                      "Y": [0.02, -0.01, -0.01, -0.03, 0.01, 0.05]}, index=idx)
    c, n = sign_concordance(r, min_obs=1)
    # 6 dias: juntos 5, opostos 1 -> (5-1)/6
    assert c.loc["X", "Y"] == pytest.approx(4 / 6) and n.loc["X", "Y"] == 6
    c2, n2 = sign_concordance(r, min_obs=1, neutral=0.001)  # dia de +0,05% vira "parado"
    assert n2.loc["X", "Y"] == 5 and c2.loc["X", "Y"] == pytest.approx(3 / 5)


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall", "ewma", "residual", "stress", "sign"])
def test_all_methods_rank_twins_first(method):
    from quantbt.correlation import MethodParams
    res = analyze(_panel(), method=method, min_obs=30, rolling_window=30, params=MethodParams(neutral=0.001))
    top = res.pairs.iloc[0]
    assert {top["a"], top["b"]} == {"A", "B"}
    assert res.pairs["corr"].between(-1, 1).all()


def test_compare_and_residual_removes_market():
    from quantbt.correlation import MethodParams
    rng = np.random.default_rng(3)
    n = 800
    mkt = rng.normal(0, 0.01, n)
    r = pd.DataFrame({
        "M": mkt,
        "P": mkt + rng.normal(0, 0.005, n),  # só mercado
        "Q": mkt + rng.normal(0, 0.005, n),  # só mercado
    }, index=pd.bdate_range("2020-01-01", periods=n))
    prices = 100 * np.exp(r.cumsum())
    res = analyze(prices, method="pearson", compare=["residual", "sign"], min_obs=30, rolling_window=0,
                  params=MethodParams(market=prices["M"], market_name="M"))
    pq = res.pairs[(res.pairs["a"].isin(["P", "Q"])) & (res.pairs["b"].isin(["P", "Q"]))].iloc[0]
    assert pq["corr"] > 0.7 and abs(pq["corr_residual"]) < 0.15  # relação vinha só do mercado
    assert pq["corr_sign"] > 0.4 and pq["mkt_a"] > 0.8



def test_coverage_counts_only_asset_lifetime():
    p = _panel()
    p.loc[p.index[:250], "D"] = np.nan  # "D" estreia no meio do período (como um ETF novo)
    p.loc[p.index[300:320], "C"] = np.nan  # "C" tem um buraco de 20 dias na própria vida
    res = analyze(p, min_obs=30, min_coverage=0.9, rolling_window=0)
    assert "D" not in res.dropped  # mais novo, mas negociou todos os dias desde a estreia
    pair = res.pairs[res.pairs[["a", "b"]].isin(["D"]).any(axis=1)]
    assert (pair["obs"] < 260).all()  # pares com D usam só o período em comum


def test_universe_kind_from_latest_and_since_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(b3, "store_dir", lambda: tmp_path)
    b3._symbol_cache.clear()
    days = pd.bdate_range("2021-01-04", periods=520)
    rows = []
    for i, d in enumerate(days):
        ds = d.strftime("%Y%m%d")
        # BDR que a B3 reclassificou de 02 para 34 no meio da história
        rows.append(_line(ds, "AAPL34", 50, bdi="02" if i < 300 else "34", volume=1e7, isin="BRAAPLBDR004"))
        if i >= 200:  # ETF lançado no meio do período
            rows.append(_line(ds, "NEWX11", 100, bdi="14", volume=1e7, isin="BRNEWXCTF000"))
    for y in sorted({d.year for d in days}):
        part = [r for r in rows if r[2:6].decode() == str(y)]
        (tmp_path / f"C{y}.TXT").write_bytes(b"\r\n".join(part))
        b3.import_file(tmp_path / f"C{y}.TXT")
    bdrs = b3.liquid_universe(0, kinds=b3.KIND_CODES["BDRs"])
    assert bdrs["ticker"].tolist() == ["AAPL34"]  # classificado pela categoria mais recente
    etfs = b3.liquid_universe(0, kinds=b3.KIND_CODES["ETFs / fundos"])
    assert etfs["ticker"].tolist() == ["NEWX11"]  # entra: negociou todos os dias desde a estreia
    strict = b3.liquid_universe(0, kinds=b3.KIND_CODES["ETFs / fundos"], since_listing=False)
    assert strict.empty  # exigindo o período inteiro, fica de fora
    b3._symbol_cache.clear()
