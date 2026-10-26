"""Eventos societários (desdobramentos, grupamentos, bonificações e dividendos) do Yahoo Finance.

Os preços do COTAHIST não são ajustados. A regra automática de b3.adjust_splits só detecta saltos
com razões típicas; aqui usamos o histórico oficial de eventos publicado no Yahoo, que também traz
bonificações pequenas (10%, 5%) e dividendos.

- Baixar: update(tickers) consulta o Yahoo e guarda em cache (~/.quantbt/eventos/), uma vez.
- Usar: load_cached(ticker) só lê o cache (nunca acessa a rede durante um backtest).
- Ajustar: apply(df, splits, dividends, total_return) ajusta OHLC; com total_return=True os
  dividendos são reinvestidos (como no Ibovespa, que é um índice de retorno total).

Limitações: o Yahoo não tem empresas que deixaram a bolsa nem cisões (ex.: GPA/Assaí em 2021);
para essas, vale a regra automática de b3.adjust_splits.
"""

from __future__ import annotations

import json
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

MAX_AGE_DAYS = 7


def cache_dir() -> Path:
    d = Path.home() / ".quantbt" / "eventos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(ticker: str) -> Path:
    return cache_dir() / f"{ticker.upper()}.json"


def fetch(ticker: str) -> dict:
    """Consulta o Yahoo. Retorna {'splits': {data: fator}, 'dividends': {data: valor}, 'ok': bool}."""
    import yfinance as yf

    from quantbt.data import _B3_LIKE

    t = ticker.upper()
    sym = f"{t}.SA" if _B3_LIKE.match(t) else t
    out = {"splits": {}, "dividends": {}, "ok": False, "fetched": time.time()}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = yf.Ticker(sym).actions
        if a is not None and len(a) == 0:
            out["ok"] = True  # ativo existe, só não tem eventos
        elif a is not None:
            idx = pd.DatetimeIndex(a.index)
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            a.index = idx.normalize()
            if "Stock Splits" in a:
                s = a["Stock Splits"][a["Stock Splits"] > 0]
                out["splits"] = {d.strftime("%Y-%m-%d"): float(v) for d, v in s.items() if v != 1}
            if "Dividends" in a:
                d = a["Dividends"][a["Dividends"] > 0]
                out["dividends"] = {k.strftime("%Y-%m-%d"): float(v) for k, v in d.groupby(level=0).sum().items()}
            out["ok"] = True
    except Exception:  # noqa: BLE001 — ativo inexistente no Yahoo: guarda vazio para não consultar de novo
        pass
    return out


def _ok(p: Path) -> bool:
    try:
        return bool(json.loads(p.read_text()).get("ok"))
    except Exception:  # noqa: BLE001
        return False


def update(tickers: Iterable[str], jobs: int = 4, force: bool = False, retry_failed: bool = False,
           log: Callable[[str], None] | None = None, should_stop: Callable[[], bool] | None = None,
           delay: float = 0.3) -> int:
    """Baixa os eventos dos ativos (em paralelo) e grava em cache. Retorna quantos têm dados no Yahoo."""
    todo = []
    for t in dict.fromkeys(x.upper().strip() for x in tickers if x.strip()):
        p = _path(t)
        stale = not p.exists() or (time.time() - p.stat().st_mtime) > MAX_AGE_DAYS * 86400
        if force or stale or (retry_failed and not _ok(p)):
            todo.append(t)
    found = 0

    def job(t: str):
        if should_stop and should_stop():
            return t, None
        time.sleep(delay)  # o Yahoo bloqueia temporariamente (HTTP 429) quem consulta rápido demais
        return t, fetch(t)

    with ThreadPoolExecutor(jobs) as ex:
        for i, (t, ev) in enumerate(ex.map(job, todo), 1):
            if ev is None:
                continue
            _path(t).write_text(json.dumps(ev))
            found += bool(ev["ok"])
            if log and (i % 50 == 0 or i == len(todo)):
                log(f"Eventos: {i:,} de {len(todo):,} consultados ({found:,} com dados no Yahoo)")
    return found


def load_cached(ticker: str) -> tuple[pd.Series, pd.Series] | None:
    """(desdobramentos, dividendos) do cache, ou None se o ativo nunca foi consultado / não existe no Yahoo."""
    p = _path(ticker)
    if not p.exists():
        return None
    try:
        ev = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not ev.get("ok"):
        return None
    sp = pd.Series({pd.Timestamp(k): v for k, v in ev.get("splits", {}).items()}, dtype=float).sort_index()
    dv = pd.Series({pd.Timestamp(k): v for k, v in ev.get("dividends", {}).items()}, dtype=float).sort_index()
    return sp, dv


def cached_count() -> int:
    return sum(1 for _ in cache_dir().glob("*.json"))


def apply(df: pd.DataFrame, splits: pd.Series, dividends: pd.Series, total_return: bool = False) -> pd.DataFrame:
    """Ajusta OHLC (índice = datas) pelos eventos. Fator do Yahoo = ações novas por ação antiga."""
    df = df.sort_index().copy()
    idx = df.index
    factor = np.ones(len(df))
    for d, f in splits.items():
        if f > 0 and idx[0] < d <= idx[-1]:
            factor[idx < d] /= f  # 8:1 -> preços antigos divididos por 8; grupamento 1:30 -> × 30
    if total_return and len(dividends):
        close_adj = df["close"].to_numpy() * factor
        for d, amount in dividends.items():
            if not (idx[0] < d <= idx[-1]):
                continue
            prev = np.searchsorted(idx, d, side="left") - 1  # último pregão antes da data ex
            if prev < 0 or not np.isfinite(close_adj[prev]) or close_adj[prev] <= 0:
                continue
            ratio = 1.0 - amount / close_adj[prev]
            if 0.5 < ratio < 1.0:  # ignora valores absurdos (erro de base ou de unidade)
                factor[: prev + 1] *= ratio
    for c in ("open", "high", "low", "close"):
        if c in df:
            df[c] = df[c] * factor
    if "quantity" in df:
        df["quantity"] = df["quantity"] / np.where(factor > 0, factor, 1.0)
    return df
