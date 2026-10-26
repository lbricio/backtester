"""Ativos externos: moedas, commodities, cripto e índices (fora do COTAHIST da B3).

Cada ativo tem um apelido curto (ex.: DOLAR, WDO, BTC, OURO) que pode ser usado na
Correlação e no Backtest junto com os códigos da B3. Os dados vêm do Yahoo Finance
(e da PTAX do Banco Central) e ficam em cache em ~/.quantbt/externos/.

Futuros da B3 (WDO, DOL, WIN, IND) não estão nos arquivos COTAHIST; os apelidos
apontam para substitutos fiéis: dólar à vista e Ibovespa. A diferença diária entre
futuro e à vista é o custo de carregamento (juros), pequena perto da variação do preço.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from quantbt.data import _yf_download, fix_yahoo_fx


@dataclass(frozen=True)
class Asset:
    alias: str
    name: str
    group: str
    source: str  # "yahoo", "bcb" ou "derived"
    symbol: str  # código no Yahoo / nº da série no BCB / expressão "A*B"
    currency: str = "USD"  # moeda de cotação ("BRL", "USD", "pts", "%")
    note: str = ""


G_FX, G_FUT, G_COM, G_CRYPTO, G_IDX = "Dólar e moedas", "Futuros B3 (substitutos)", "Commodities", "Cripto", "Índices"

_ASSETS = [
    # moedas
    Asset("DOLAR", "Dólar à vista (USD/BRL)", G_FX, "yahoo", "BRL=X", "BRL"),
    Asset("PTAX", "Dólar PTAX venda (Banco Central)", G_FX, "bcb", "1", "BRL", "fixing oficial ~13h"),
    Asset("EURO", "Euro (EUR/BRL)", G_FX, "yahoo", "EURBRL=X", "BRL"),
    Asset("DXY", "Índice do dólar (DXY)", G_FX, "yahoo", "DX-Y.NYB", "pts"),
    Asset("YUAN", "Dólar/yuan (USD/CNY)", G_FX, "yahoo", "CNY=X", "pts"),
    # futuros da B3 -> substitutos
    Asset("WDO", "Mini dólar (substituto: dólar à vista)", G_FUT, "yahoo", "BRL=X", "BRL",
          "futuro WDO não está no COTAHIST; usa USD/BRL à vista"),
    Asset("DOL", "Dólar cheio (substituto: dólar à vista)", G_FUT, "yahoo", "BRL=X", "BRL",
          "futuro DOL não está no COTAHIST; usa USD/BRL à vista"),
    Asset("WIN", "Mini índice (substituto: Ibovespa)", G_FUT, "yahoo", "^BVSP", "pts",
          "futuro WIN não está no COTAHIST; usa o Ibovespa"),
    Asset("IND", "Índice cheio (substituto: Ibovespa)", G_FUT, "yahoo", "^BVSP", "pts",
          "futuro IND não está no COTAHIST; usa o Ibovespa"),
    # commodities (futuros contínuos, 1º vencimento)
    Asset("PETROLEO", "Petróleo WTI", G_COM, "yahoo", "CL=F"),
    Asset("BRENT", "Petróleo Brent", G_COM, "yahoo", "BZ=F"),
    Asset("GAS", "Gás natural", G_COM, "yahoo", "NG=F"),
    Asset("OURO", "Ouro", G_COM, "yahoo", "GC=F"),
    Asset("PRATA", "Prata", G_COM, "yahoo", "SI=F"),
    Asset("PLATINA", "Platina", G_COM, "yahoo", "PL=F"),
    Asset("COBRE", "Cobre", G_COM, "yahoo", "HG=F"),
    Asset("ALUMINIO", "Alumínio", G_COM, "yahoo", "ALI=F"),
    Asset("MINERIO", "Minério de ferro (SGX 62%)", G_COM, "yahoo", "TIO=F"),
    Asset("SOJA", "Soja (CBOT)", G_COM, "yahoo", "ZS=F"),
    Asset("MILHO", "Milho (CBOT)", G_COM, "yahoo", "ZC=F"),
    Asset("TRIGO", "Trigo (CBOT)", G_COM, "yahoo", "ZW=F"),
    Asset("CAFE", "Café arábica (ICE)", G_COM, "yahoo", "KC=F"),
    Asset("ACUCAR", "Açúcar (ICE)", G_COM, "yahoo", "SB=F"),
    Asset("ALGODAO", "Algodão (ICE)", G_COM, "yahoo", "CT=F"),
    Asset("CACAU", "Cacau (ICE)", G_COM, "yahoo", "CC=F"),
    Asset("BOI", "Boi gordo (CME)", G_COM, "yahoo", "LE=F"),
    # cripto
    Asset("BTC", "Bitcoin (US$)", G_CRYPTO, "yahoo", "BTC-USD"),
    Asset("BTCBRL", "Bitcoin (R$)", G_CRYPTO, "derived", "BTC*DOLAR", "BRL"),
    Asset("ETH", "Ethereum (US$)", G_CRYPTO, "yahoo", "ETH-USD"),
    Asset("ETHBRL", "Ethereum (R$)", G_CRYPTO, "derived", "ETH*DOLAR", "BRL"),
    # índices
    Asset("IBOV", "Ibovespa", G_IDX, "yahoo", "^BVSP", "pts"),
    Asset("SP500", "S&P 500", G_IDX, "yahoo", "^GSPC", "USD"),
    Asset("NASDAQ", "Nasdaq Composite", G_IDX, "yahoo", "^IXIC", "USD"),
    Asset("VIX", "VIX (volatilidade S&P)", G_IDX, "yahoo", "^VIX", "pts"),
    Asset("US10Y", "Juro EUA 10 anos (taxa)", G_IDX, "yahoo", "^TNX", "%"),
]
CATALOG: dict[str, Asset] = {a.alias: a for a in _ASSETS}
GROUPS: dict[str, list[str]] = {}
for _a in _ASSETS:
    GROUPS.setdefault(_a.group, []).append(_a.alias)
# grupos sugeridos para "adicionar à lista" (sem repetir séries idênticas como WDO/DOL/DOLAR)
PRESETS: dict[str, list[str]] = {
    "Dólar, mini dólar e moedas": ["DOLAR", "WDO", "EURO", "DXY"],
    "Commodities": ["PETROLEO", "BRENT", "OURO", "PRATA", "COBRE", "MINERIO", "SOJA", "MILHO", "CAFE", "ACUCAR", "BOI"],
    "Bitcoin e cripto": ["BTC", "BTCBRL", "ETH"],
    "Índices e mini índice": ["IBOV", "WIN", "SP500", "NASDAQ", "VIX"],
    "Tudo (macro + commodities + cripto)": ["DOLAR", "EURO", "DXY", "PETROLEO", "BRENT", "GAS", "OURO", "PRATA",
                                             "COBRE", "MINERIO", "SOJA", "MILHO", "TRIGO", "CAFE", "ACUCAR",
                                             "BOI", "BTC", "ETH", "IBOV", "SP500", "VIX", "US10Y"],
}

MAX_AGE_HOURS = 12  # cache mais velho que isso é atualizado automaticamente


def is_external(code: str) -> bool:
    return code.upper().strip() in CATALOG


def same_underlying(a: str, b: str) -> bool:
    """Dois apelidos que usam a mesma série (ex.: DOLAR, WDO e DOL; IBOV e WIN)."""
    x, y = CATALOG.get(a.upper()), CATALOG.get(b.upper())
    return bool(x and y and x.source == y.source and x.symbol == y.symbol)


def describe(code: str) -> str:
    a = CATALOG.get(code.upper().strip())
    return a.name if a else ""


def cache_dir() -> Path:
    d = Path.home() / ".quantbt" / "externos"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(key: str) -> Path:
    safe = "".join(ch if ch.isalnum() else "_" for ch in key)
    return cache_dir() / f"{safe}.parquet"


# ---------------------------------------------------------------- download
def _yahoo(symbol: str, start: str) -> pd.DataFrame:
    raw = _yf_download(symbol, start=start, auto_adjust=True, progress=False, multi_level_index=False)
    if raw is None or raw.empty:
        raise ValueError(f"Yahoo não retornou dados para {symbol}.")
    raw.columns = [str(c).lower() for c in raw.columns]
    df = raw[["open", "high", "low", "close"]].copy()
    df["volume"] = raw["volume"] if "volume" in raw else 0.0
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.DatetimeIndex(df.index).normalize()
    df.index.name = "date"
    return fix_yahoo_fx(df.dropna(subset=["close"]), symbol)


def _bcb(series: str, start: str) -> pd.DataFrame:
    """Série diária do SGS/Banco Central (consulta em blocos: limite de 10 anos por pedido)."""
    d0 = pd.Timestamp(start).date()
    today = date.today()
    rows = []
    while d0 <= today:
        d1 = min(d0 + timedelta(days=365 * 5), today)
        url = (f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series}/dados?formato=json"
               f"&dataInicial={d0:%d/%m/%Y}&dataFinal={d1:%d/%m/%Y}")
        req = urllib.request.Request(url, headers={"User-Agent": "quantbt"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                rows += json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code != 404:  # 404 = intervalo sem cotações (ex.: só fim de semana)
                raise
        d0 = d1 + timedelta(days=1)
    if not rows:
        raise ValueError(f"Banco Central não retornou dados para a série {series}.")
    s = pd.Series({pd.to_datetime(r["data"], dayfirst=True): float(r["valor"]) for r in rows}).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    df = pd.DataFrame({"open": s, "high": s, "low": s, "close": s, "volume": 0.0})
    df.index.name = "date"
    return df


def _fetch(asset: Asset, start: str) -> pd.DataFrame:
    if asset.source == "yahoo":
        return _yahoo(asset.symbol, start)
    if asset.source == "bcb":
        return _bcb(asset.symbol, start)
    if asset.source == "derived":
        a, b = asset.symbol.split("*")
        x, y = load(a, start), load(b, start)
        y = y["close"].reindex(x.index.union(y.index)).ffill().reindex(x.index)
        out = x[["open", "high", "low", "close"]].mul(y, axis=0)
        out["volume"] = 0.0
        return out.dropna()
    raise ValueError(f"Fonte desconhecida: {asset.source}")


def _key(asset: Asset) -> str:
    return asset.symbol if asset.source != "derived" else asset.alias


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".json")


def _save_cache(df: pd.DataFrame, path: Path, requested_start: str) -> None:
    """Grava o cache de forma atômica, junto com o início pedido no download."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    _meta_path(path).write_text(json.dumps({"start": requested_start}))


def _cached_start(path: Path, df: pd.DataFrame) -> pd.Timestamp:
    """Início coberto pelo cache: o pedido no download (o ativo pode nem existir antes disso)."""
    try:
        return pd.Timestamp(json.loads(_meta_path(path).read_text())["start"])
    except Exception:  # noqa: BLE001 — cache antigo sem metadados
        return df.index[0] if len(df) else pd.Timestamp.max


def load(code: str, start: str | None = None, end: str | None = None, refresh: bool = False,
         strict: bool = False) -> pd.DataFrame:
    """OHLCV diário de um ativo externo (formato do backtester), usando o cache local.

    O cache é atualizado se tiver mais de MAX_AGE_HOURS ou não cobrir o início pedido. Se a
    atualização falhar (sem internet, fonte fora do ar), usa o cache existente — a menos que
    strict=True (usado na atualização explícita, que precisa relatar a falha).
    """
    asset = CATALOG.get(code.upper().strip())
    if asset is None:
        raise ValueError(f"'{code}' não está no catálogo de ativos externos.")
    start = start or str(date.today().replace(year=date.today().year - 10))
    path = _cache_path(_key(asset))
    cached = pd.read_parquet(path) if path.exists() else None
    need = refresh or cached is None
    if not need:
        fresh = (time.time() - path.stat().st_mtime) < MAX_AGE_HOURS * 3600
        covers = _cached_start(path, cached) <= pd.Timestamp(start) + pd.Timedelta(days=7)
        need = not (fresh and covers)
    df = cached
    if need:
        fetch_start = start
        if cached is not None and len(cached):  # mantém o histórico mais antigo já baixado
            fetch_start = str(min(_cached_start(path, cached), pd.Timestamp(start)).date())
        try:
            df = _fetch(asset, fetch_start)
            _save_cache(df, path, fetch_start)
        except Exception:
            if cached is None or strict:
                raise
            df = cached  # sem conexão: segue com o que já existe
    df = df.sort_index()
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    return df.astype(float)


def load_closes(codes: Iterable[str], start=None, end=None, in_brl: bool = False) -> pd.DataFrame:
    """Painel de fechamentos dos ativos externos. in_brl: converte cotações em US$ para R$."""
    codes = [c.upper().strip() for c in codes]
    cols, usd = {}, None
    for c in codes:
        s = load(c, start, end)["close"]
        if in_brl and CATALOG[c].currency == "USD":
            if usd is None:
                usd = load("DOLAR", start, end)["close"]
            s = s * usd.reindex(s.index.union(usd.index)).ffill().reindex(s.index)
        cols[c] = s
    return pd.DataFrame(cols).sort_index()


def update_all(codes: Iterable[str] | None = None, start: str | None = None,
               log: Callable[[str], None] | None = None) -> list[str]:
    """Baixa/atualiza o cache dos ativos do catálogo."""
    done, seen = [], set()
    for c in (codes or list(CATALOG)):
        a = CATALOG[c]
        if _key(a) in seen:
            continue
        seen.add(_key(a))
        try:
            df = load(c, start, refresh=True, strict=True)
            done.append(c)
            if log:
                log(f"{c}: {len(df):,} dias ({df.index[0]:%d/%m/%Y} → {df.index[-1]:%d/%m/%Y})")
        except Exception as e:  # noqa: BLE001
            if log:
                log(f"{c}: falhou ({e})")
    return done


def align_to(panel_index: pd.DatetimeIndex, ext: pd.DataFrame, max_gap: int = 3) -> pd.DataFrame:
    """Alinha séries externas ao calendário de pregões da B3.

    Usa o último preço disponível até cada pregão (no máximo `max_gap` dias corridos de
    defasagem), assim cripto (24/7) e mercados com feriados diferentes ficam comparáveis.
    """
    if ext.empty:
        return ext.reindex(panel_index)
    ext = ext.sort_index()
    # para cada coluna, último valor válido até a data, com tolerância em dias corridos
    out = {c: ext[c].dropna().reindex(panel_index, method="ffill", tolerance=pd.Timedelta(days=max_gap))
           for c in ext.columns}
    return pd.DataFrame(out, index=panel_index)


def b3_calendar(start=None, end=None) -> pd.DatetimeIndex:
    """Pregões da B3 no período (a partir do Ibovespa; dias úteis se indisponível)."""
    try:
        idx = load("IBOV", start, end).index
        if len(idx):
            return idx
    except Exception:  # noqa: BLE001
        pass
    return pd.bdate_range(start or date.today().replace(year=date.today().year - 10), end or date.today())


def split_codes(codes: Iterable[str]) -> tuple[list[str], list[str]]:
    """Separa códigos da B3 dos apelidos de ativos externos."""
    b3c, ext = [], []
    for c in codes:
        (ext if is_external(c) else b3c).append(c.upper().strip())
    return b3c, ext


def combined_closes(codes: Iterable[str], b3_loader: Callable[[list[str]], pd.DataFrame],
                    start=None, end=None, in_brl: bool = False) -> pd.DataFrame:
    """Painel com ativos da B3 + externos, alinhado ao calendário da B3."""
    codes = list(dict.fromkeys(c.upper().strip() for c in codes if c.strip()))
    b3c, ext = split_codes(codes)
    parts = []
    if b3c:
        try:
            parts.append(b3_loader(b3c))
        except ValueError:
            if not ext:
                raise
    base = parts[0].index if parts and len(parts[0]) else b3_calendar(start, end)
    if ext:
        parts.append(align_to(pd.DatetimeIndex(base), load_closes(ext, start, end, in_brl)))
    if not parts:
        raise ValueError("Nenhum ativo encontrado.")
    panel = pd.concat([p.reindex(base) for p in parts], axis=1)
    return panel[[c for c in codes if c in panel.columns]]


def catalog_table() -> pd.DataFrame:
    return pd.DataFrame(
        [(a.alias, a.name, a.group, a.currency, a.source, a.symbol, a.note) for a in _ASSETS],
        columns=["apelido", "nome", "grupo", "moeda", "fonte", "código", "obs"],
    )

