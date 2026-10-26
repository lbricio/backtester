"""Carregamento unificado de preços: base B3, ativos externos (DOLAR, BTC, OURO...) ou Yahoo."""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pandas as pd

from quantbt import b3, external
from quantbt import data as data_mod

SRC_B3, SRC_YAHOO = "Base B3 (COTAHIST)", "Yahoo Finance"


def load_prices(ticker: str, source: str = SRC_B3, start: str | None = None, end: str | None = None,
                total_return: bool = False) -> pd.DataFrame:
    """OHLCV diário. Na fonte Base B3, apelidos do catálogo externo também funcionam."""
    t = ticker.strip()
    if not t:
        raise ValueError("Informe um ativo.")
    if source == SRC_B3:
        if external.is_external(t):
            return external.load(t.upper(), start, end)
        return b3.load_ticker(t.upper().removesuffix(".SA"), start, end, total_return=total_return)
    return data_mod.load_yahoo(t, start, end)


def describe(ticker: str) -> str:
    return external.describe(ticker) or ticker.upper()


def current_cdi() -> tuple[float, str]:
    """CDI anualizado mais recente (Banco Central, série 4389). Retorna (taxa em fração, data)."""
    url = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.4389/dados/ultimos/1?formato=json"
    req = urllib.request.Request(url, headers={"User-Agent": "quantbt"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        row = json.loads(resp.read().decode())[-1]
    return float(row["valor"]) / 100, row["data"]


def load_many(tickers, start: str | None = None, end: str | None = None, total_return: bool = False
              ) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """OHLCV de vários ativos (base B3 + externos). Retorna (dados por ativo, não encontrados)."""
    codes = list(dict.fromkeys(t.upper().strip() for t in tickers if t.strip()))
    b3c, ext = external.split_codes(codes)
    data: dict[str, pd.DataFrame] = {}
    if b3c:
        try:
            data.update(b3.load_ohlcv_many(b3c, start, end, total_return=total_return))
        except FileNotFoundError:
            pass
    for c in ext:
        try:
            data[c] = external.load(c, start, end)
        except Exception:  # noqa: BLE001
            pass
    missing = [c for c in codes if c not in data]
    return {c: data[c] for c in codes if c in data}, missing


def cdi_index(start: str | None = None) -> pd.Series:
    """Índice do CDI acumulado dia a dia (Banco Central, série 12), em cache por 12 h.

    Serve para comparar uma estratégia com o CDI que realmente rendeu nas mesmas datas.
    """
    path = external.cache_dir() / "CDI_diario.parquet"
    start = start or str(pd.Timestamp.today().date() - pd.DateOffset(years=11))
    df = pd.read_parquet(path) if path.exists() else None
    fresh = df is not None and (time.time() - path.stat().st_mtime) < external.MAX_AGE_HOURS * 3600
    if df is None or not fresh or df.index[0] > pd.Timestamp(start) + pd.Timedelta(days=7):
        try:
            fetch_from = str(min(pd.Timestamp(start), df.index[0]).date()) if df is not None else start
            df = external._bcb("12", fetch_from)[["close"]]
            df.to_parquet(path)
        except Exception:
            if df is None:
                raise
    rate = df["close"] / 100.0  # % ao dia -> fração
    idx = np.cumprod(1.0 + rate)
    return idx[idx.index >= pd.Timestamp(start)] if start else idx


def cdi_cagr_between(cdi: pd.Series, dates: pd.DatetimeIndex, ppy: float) -> float:
    """CDI anualizado entre a primeira e a última data, na mesma convenção de anos do backtest."""
    if cdi is None or len(cdi) == 0 or len(dates) < 2:
        return float("nan")
    s = cdi.reindex(cdi.index.union(dates)).ffill().reindex(dates)
    if s.isna().iloc[0] or s.isna().iloc[-1]:
        return float("nan")
    years = (len(dates) - 1) / ppy
    return float((s.iloc[-1] / s.iloc[0]) ** (1 / max(years, 1e-9)) - 1)
