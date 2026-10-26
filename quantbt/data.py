"""Carregamento de dados OHLCV."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_B3_LIKE = re.compile(r"^[A-Z]{4}\d{1,2}[A-Z]?$")


def fix_yahoo_fx(df: pd.DataFrame | pd.Series, symbol: str):
    """Câmbio no Yahoo (códigos '=X'): o candle datado D+1 reflete o fechamento do dia D.

    Verificado contra a PTAX e o Ibovespa (correlação diária IBOV × dólar vai de ~0 para ~-0,5).
    """
    if str(symbol).upper().endswith("=X"):
        return df.shift(-1).iloc[:-1]
    return df


def _yf_download(*args, **kwargs):
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("yfinance não está instalado. Rode: uv sync --extra yahoo") from e
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # yfinance emite avisos de depreciação do pandas
        return yf.download(*args, **kwargs)

_ALIASES = {
    "date": "date", "datetime": "date", "time": "date", "timestamp": "date", "data": "date",
    "open": "open", "abertura": "open", "o": "open",
    "high": "high", "max": "high", "maxima": "high", "máxima": "high", "h": "high",
    "low": "low", "min": "low", "minima": "low", "mínima": "low", "l": "low",
    "close": "close", "fechamento": "close", "c": "close", "adj close": "adj_close",
    "volume": "volume", "vol": "volume", "v": "volume",
}


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza colunas (open/high/low/close/volume), índice de datas e ordenação."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):  # formato do yfinance
        df.columns = df.columns.get_level_values(0)
    df.columns = [_ALIASES.get(str(c).strip().lower(), str(c).strip().lower()) for c in df.columns]
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], utc=False, errors="coerce")
        df = df.set_index("date")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df = df[~df.index.isna()]

    if "close" not in df.columns and "adj_close" in df.columns:
        df["close"] = df["adj_close"]
    if "close" not in df.columns:
        raise ValueError("Os dados precisam ter ao menos a coluna 'close'.")
    for col in ("open", "high", "low"):
        if col not in df.columns:
            df[col] = df["close"]
    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["close"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "date"
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    # tenta detectar separador ( , ou ; ) automaticamente
    raw = pd.read_csv(path, sep=None, engine="python")
    return normalize(raw)


def load_yahoo(ticker: str, start: str | None = None, end: str | None = None, interval: str = "1d") -> pd.DataFrame:
    raw = _yf_download(ticker, start=start, end=end, interval=interval,
                       auto_adjust=True, progress=False, multi_level_index=False)
    if raw is None or raw.empty:
        raise ValueError(f"Nenhum dado retornado para '{ticker}'.")
    df = normalize(raw)
    return fix_yahoo_fx(df, ticker) if interval in ("1d", "1wk", "1mo") else df


def sample_data(n: int = 2500, seed: int = 42, start: str = "2015-01-01") -> pd.DataFrame:
    """Gera série sintética (passeio aleatório com regimes de tendência) para testes offline."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    regime = np.repeat(rng.normal(0.0003, 0.001, size=n // 120 + 1), 120)[:n]
    vol = 0.012 + 0.008 * np.abs(np.sin(np.arange(n) / 180))
    rets = regime + rng.normal(0, 1, n) * vol
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    hi = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.006, n)))
    lo = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.006, n)))
    volume = rng.integers(1e5, 1e6, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": hi, "low": lo, "close": close, "volume": volume},
        index=pd.DatetimeIndex(idx, name="date"),
    )


def load_yahoo_closes(tickers: list[str], start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Painel de fechamentos ajustados (datas x ativos). Códigos da B3 sem sufixo ganham '.SA'."""
    mapping = {(f"{t}.SA" if _B3_LIKE.match(t) else t): t for t in tickers}
    raw = _yf_download(list(mapping), start=start, end=end, auto_adjust=True, progress=False,
                       group_by="column", multi_level_index=True)
    if raw is None or raw.empty:
        raise ValueError("Nenhum dado retornado pelo Yahoo Finance.")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].set_axis([next(iter(mapping))], axis=1)
    close = close.rename(columns=mapping).dropna(how="all", axis=1)
    if getattr(close.index, "tz", None) is not None:
        close.index = close.index.tz_localize(None)
    for sym, t in mapping.items():  # câmbio: corrige a data coluna a coluna
        if t in close and sym.upper().endswith("=X"):
            close[t] = close[t].shift(-1)
    return close[[t for t in tickers if t in close.columns]]
