"""Séries históricas oficiais da B3 (arquivos COTAHIST).

Os arquivos anuais ficam em
https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ANO}.ZIP
e têm layout de largura fixa (245 caracteres por linha). Este módulo baixa,
converte e guarda cada ano em Parquet em ~/.quantbt/b3/, mantendo só o mercado
à vista (ações, units, FIIs, Fiagro, ETFs e BDRs, inclusive empresas em recuperação).
Direitos e recibos de subscrição são descartados na leitura.

Séries de papéis que trocaram de código são unidas pelo ISIN (ex.: 'PET 4' -> PETR4).
Antes de 1994 os preços são convertidos para reais (cortes de zeros das moedas antigas).

Atenção: os preços do COTAHIST NÃO são ajustados por proventos. Desdobramentos e
grupamentos geram saltos artificiais; `adjust_splits` corrige os casos
detectáveis (razões típicas como 2:1, 1:10). Dividendos não são ajustados.
"""

from __future__ import annotations

import io
import os
import tempfile
import urllib.request
import zipfile
from datetime import date
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

URL = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{year}.ZIP"
FIRST_YEAR = 1986

# CODBDI (categoria do papel) mantidos. Descartados: 10 direitos/recibos, 22 bônus, 96 fracionário.
KIND_CODES: dict[str, tuple[str, ...]] = {
    "Ações / units": ("02", "07", "08", "58"),  # 07/08 = recuperação extrajudicial/judicial
    "BDRs": ("34", "35", "36"),
    "FIIs / Fiagro": ("12", "13"),
    "ETFs / fundos": ("14",),
}
KEEP_BDI = {c for codes in KIND_CODES.values() for c in codes}
KEEP_TPMERC = {"010"}  # mercado à vista

COLUMNS = ["date", "ticker", "name", "spec", "bdi", "open", "high", "low", "close",
           "avg", "trades", "quantity", "volume", "isin"]


def store_dir() -> Path:
    d = Path.home() / ".quantbt" / "b3"
    d.mkdir(parents=True, exist_ok=True)
    return d


def year_path(year: int) -> Path:
    return store_dir() / f"cotahist_{year}.parquet"


def _write_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    """Grava num arquivo temporário e troca no fim: nunca deixa um ano pela metade."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem + ".", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def available_years() -> list[int]:
    out = []
    for p in store_dir().glob("cotahist_*.parquet"):
        try:
            out.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(out)


# ---------------------------------------------------------------- parser
def _num(b: bytes) -> float:
    return int(b) / 100.0


def parse_cotahist(raw: bytes | Iterable[bytes]) -> pd.DataFrame:
    """Converte o conteúdo de um arquivo COTAHIST (texto) em DataFrame."""
    lines = raw.splitlines() if isinstance(raw, (bytes, bytearray)) else raw
    rows = []
    for ln in lines:
        if len(ln) < 245 or ln[0:2] != b"01":
            continue
        bdi = ln[10:12].decode()
        if bdi not in KEEP_BDI or ln[24:27].decode() not in KEEP_TPMERC:
            continue
        fat = int(ln[210:217]) or 1  # fator de cotação (preço por 1 ou por 1000 ações)
        rows.append((
            ln[2:10].decode(),
            ln[12:24].decode("latin-1").strip(),
            ln[27:39].decode("latin-1").strip(),
            ln[39:49].decode("latin-1").strip(),
            bdi,
            _num(ln[56:69]) / fat,
            _num(ln[69:82]) / fat,
            _num(ln[82:95]) / fat,
            _num(ln[108:121]) / fat,
            _num(ln[95:108]) / fat,
            int(ln[147:152]),
            int(ln[152:170]),
            _num(ln[170:188]),
            ln[230:242].decode("latin-1").strip(),
        ))
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in ("ticker", "name", "spec", "bdi", "isin"):
        df[c] = df[c].astype("category")
    return df


def _read_any(path_or_bytes: Path | bytes) -> bytes:
    """Aceita .ZIP ou .TXT do COTAHIST e devolve o texto em bytes."""
    data = Path(path_or_bytes).read_bytes() if isinstance(path_or_bytes, (str, Path)) else path_or_bytes
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = z.namelist()
            name = next((n for n in names if n.upper().endswith(".TXT")), names[0])
            return z.read(name)
    return data


def _year_of(df: pd.DataFrame, fallback: int | None) -> int:
    if len(df):
        return int(df["date"].dt.year.mode().iloc[0])
    if fallback is None:
        raise ValueError("Arquivo COTAHIST vazio ou inválido.")
    return fallback


def import_file(path: str | Path) -> tuple[int, int]:
    """Importa um COTAHIST baixado manualmente. Retorna (ano, nº de linhas)."""
    df = parse_cotahist(_read_any(Path(path)))
    year = _year_of(df, None)
    _write_parquet(df, year_path(year), index=False)
    return year, len(df)


def download_year(year: int, progress: Callable[[int, int], None] | None = None, timeout: int = 120) -> int:
    """Baixa e armazena um ano. Retorna o nº de linhas."""
    req = urllib.request.Request(URL.format(year=year), headers={"User-Agent": "Mozilla/5.0 quantbt"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        buf = bytearray()
        while chunk := resp.read(1 << 20):
            buf.extend(chunk)
            if progress:
                progress(len(buf), total)
    df = parse_cotahist(_read_any(bytes(buf)))
    _write_parquet(df, year_path(_year_of(df, year)), index=False)
    return len(df)


def download_years(
    years: Iterable[int],
    log: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[int]:
    done = []
    for y in years:
        if should_stop and should_stop():
            break
        if log:
            log(f"Baixando {y}…")
        try:
            n = download_year(y)
            done.append(y)
            if log:
                log(f"{y}: {n:,} registros salvos.")
        except Exception as e:
            if log:
                log(f"{y}: falhou ({e}).")
    return done


# ---------------------------------------------------------------- consulta
# especificações de direitos/recibos de subscrição (FIIs como ALZR12, bancos como BPAC13):
# duram poucos dias, ganham ISIN novo a cada rodada e não são ativos de verdade
_RIGHTS_RE = r"^(?:DIR|REC|RCB|BNS)\b"


def load_store(years: Iterable[int] | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    ys = [y for y in (years or available_years()) if year_path(y).exists()]
    if not ys:
        raise FileNotFoundError("Base B3 vazia. Baixe ou importe arquivos COTAHIST na aba 'Base B3'.")
    read_cols = None if columns is None else list(dict.fromkeys([*columns, "spec"]))
    frames = [pd.read_parquet(year_path(y), columns=read_cols) for y in ys]
    df = pd.concat(frames, ignore_index=True)
    for c in ("ticker", "name", "spec", "bdi", "isin"):
        if c in df:
            df[c] = df[c].astype(str)
    df = df[~df["spec"].str.match(_RIGHTS_RE)]
    if columns is not None and "spec" not in columns:
        df = df.drop(columns="spec")
    return df.reset_index(drop=True)


def _years_between(start, end) -> list[int]:
    y0 = pd.Timestamp(start).year if start else FIRST_YEAR
    y1 = pd.Timestamp(end).year if end else date.today().year
    return [y for y in available_years() if y0 <= y <= y1]


# ---------------------------------------------------------------- moeda e eventos
# Cortes de zeros nas trocas de moeda (data do 1º pregão na moeda nova, divisor).
# Antes de 1994 os arquivos trazem preços na moeda da época; aqui tudo vira R$.
CURRENCY_CUTS = [
    ("1986-03-04", 1000.0),   # Cruzeiro -> Cruzado
    ("1987-06-01", 0.001),    # mudança de convenção de cotação observada nos arquivos (salto de ×1000)
    ("1989-01-18", 1000.0),   # Cruzado -> Cruzado Novo
    ("1993-08-02", 1000.0),   # Cruzeiro -> Cruzeiro Real
    ("1994-07-04", 2750.0),   # Cruzeiro Real -> Real
]
REAL_PLAN = pd.Timestamp("1994-07-04")
PRICE_COLS = ("open", "high", "low", "close", "avg", "volume")


def to_reais(df: pd.DataFrame, date_col: str | None = "date") -> pd.DataFrame:
    """Converte preços/volumes anteriores ao Plano Real para reais."""
    dates = df[date_col] if date_col else df.index.to_series()
    if len(df) == 0 or dates.min() >= REAL_PLAN:
        return df
    cuts = np.array([np.datetime64(d) for d, _ in CURRENCY_CUTS])
    div = np.array([f for _, f in CURRENCY_CUTS])
    # divisor acumulado para datas antes de cada corte
    cum = np.r_[np.cumprod(div[::-1])[::-1], 1.0]
    k = np.searchsorted(cuts, dates.to_numpy(dtype="datetime64[ns]"), side="right")
    factor = cum[k]
    df = df.copy()
    for c in PRICE_COLS:
        if c in df:
            df[c] = df[c] / factor
    return df


_ISIN_RE = r"^BR[A-Z0-9]{10}$"
_symbol_cache: dict = {}


def build_symbol_map(df: pd.DataFrame) -> pd.Series:
    """ISIN -> código atual, a partir de linhas (date, ticker, isin).

    Junta séries de papéis que trocaram de código (ex.: 'PET 4' -> PETR4). Se um código foi
    reaproveitado por outro papel (ISIN diferente, ex.: FIXX11 FII até 2019 e ETF desde 2025),
    o papel mais recente fica com o código e o antigo vira 'FIXX11~2019', evitando emendar
    as duas histórias.
    """
    df = df[df["isin"].str.match(_ISIN_RE)]
    last = (df.sort_values("date").groupby("isin")
            .agg(ticker=("ticker", "last"), last_date=("date", "max")))
    newest = last.groupby("ticker")["last_date"].transform("max")
    old = last["last_date"] < newest
    last.loc[old, "ticker"] = last.loc[old, "ticker"] + "~" + last.loc[old, "last_date"].dt.strftime("%Y")
    dup = old & last["ticker"].duplicated(keep=False)  # dois papéis antigos encerrados no mesmo ano
    last.loc[dup, "ticker"] = last.loc[dup, "ticker"] + "-" + last.loc[dup, "last_date"].dt.strftime("%m")
    return last["ticker"]


def symbol_map() -> pd.Series:
    """ISIN -> código atual de toda a base (em cache enquanto os arquivos não mudarem)."""
    sig = tuple((y, year_path(y).stat().st_mtime) for y in available_years())
    if _symbol_cache.get("sig") != sig:
        _symbol_cache.update(sig=sig, map=build_symbol_map(load_store(columns=["date", "ticker", "isin"])))
    return _symbol_cache["map"]


def with_symbols(df: pd.DataFrame) -> pd.DataFrame:
    """Substitui 'ticker' pelo código atual do papel (via ISIN) e remove datas duplicadas."""
    if "isin" not in df:
        return df
    sym = df["isin"].map(symbol_map())
    df = df.assign(ticker=sym.where(sym.notna(), df["ticker"]))
    if "volume" in df:
        df = df.sort_values("volume")
    return df.drop_duplicates(["ticker", "date"], keep="last")


_SPLIT_N = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50, 100)
SPLIT_RATIOS = np.array([1 / r for r in _SPLIT_N] + list(_SPLIT_N) + [2 / 3, 3 / 2])


def adjust_splits(df: pd.DataFrame, tol: float = 0.03, min_jump: float = 0.3, tol_big: float = 0.08) -> pd.DataFrame:
    """Corrige desdobramentos, grupamentos e bonificações de 50% detectados por saltos típicos.

    Um salto abertura(t)/fechamento(t-1) maior que `min_jump` e próximo (±tol) de uma
    razão típica (2:1, 1:10, 3:2...) é tratado como evento corporativo e o histórico
    anterior é reescalado. Bonificações pequenas (10%, 20%) e dividendos não são detectáveis.
    """
    df = df.copy()
    if len(df) < 2:
        return df
    ratio = (df["open"] / df["close"].shift(1)).to_numpy()
    factor = np.ones(len(df))
    big = np.abs(np.log(np.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0))) > np.log(1 + min_jump)
    for i in np.where(big)[0]:
        r = ratio[i]
        # razões grandes (≥ 3:1) não se confundem com quedas reais: aceita mais oscilação no dia do evento
        t = np.where((SPLIT_RATIOS >= 3) | (SPLIT_RATIOS <= 1 / 3), tol_big, tol)
        near = SPLIT_RATIOS[np.abs(r / SPLIT_RATIOS - 1) < t]
        if len(near):
            factor[:i] *= near[0]
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * factor
    if "quantity" in df:
        df["quantity"] = df["quantity"] / factor
    return df


def _prepare(df: pd.DataFrame, adjust: bool, ticker: str | None = None, total_return: bool = False) -> pd.DataFrame:
    """Série de um papel: datas únicas, em reais, com eventos corrigidos.

    Com adjust: usa os eventos do Yahoo em cache (desdobramentos, grupamentos, bonificações e, com
    total_return, dividendos reinvestidos) e depois a regra automática para o que faltar.
    """
    from quantbt import events

    g = df.drop_duplicates("date", keep="last").set_index("date").sort_index()
    g = to_reais(g, date_col=None)
    if not adjust:
        return g
    ev = events.load_cached(ticker) if ticker else None
    if ev is not None:
        g = events.apply(g, ev[0], ev[1], total_return=total_return)
    return adjust_splits(g)


def load_ticker(ticker: str, start=None, end=None, adjust: bool = True, total_return: bool = False) -> pd.DataFrame:
    """OHLCV diário de um ativo da base local (formato do backtester), em reais."""
    years = _years_between(start, end)
    df = load_store(years, ["date", "ticker", "isin", "open", "high", "low", "close", "volume", "quantity"])
    df = with_symbols(df)
    t = ticker.upper().strip()
    df = df[df["ticker"] == t]
    if df.empty:
        raise ValueError(f"'{ticker}' não encontrado na base B3 ({years[0] if years else '?'}–"
                         f"{years[-1] if years else '?'}).")
    df = _prepare(df, adjust, t, total_return)
    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    out = df[["open", "high", "low", "close", "volume"]].astype(float)
    out.index.name = "date"
    return out


def load_closes(tickers: Iterable[str], start=None, end=None, adjust: bool = True,
                total_return: bool = False) -> pd.DataFrame:
    """Painel de fechamentos em reais (datas x ativos)."""
    tickers = list(dict.fromkeys(t.upper().strip() for t in tickers if t.strip()))
    years = _years_between(start, end)
    df = load_store(years, ["date", "ticker", "isin", "open", "high", "low", "close", "volume", "quantity"])
    df = with_symbols(df)
    df = df[df["ticker"].isin(tickers)]
    cols = {t: _prepare(g, adjust, t, total_return)["close"] for t, g in df.groupby("ticker", sort=False)}
    if not cols:
        raise ValueError("Nenhum dos ativos foi encontrado na base B3.")
    panel = pd.DataFrame(cols).sort_index()
    if start:
        panel = panel[panel.index >= pd.Timestamp(start)]
    if end:
        panel = panel[panel.index <= pd.Timestamp(end)]
    return panel[[t for t in tickers if t in panel.columns]]


def kind_of(bdi: str) -> str:
    return next((k for k, codes in KIND_CODES.items() if bdi in codes), "Outros")


UNIVERSE_COLUMNS = ["ticker", "name", "kind", "first", "days", "sessions", "avg_volume"]


def liquid_universe(
    n: int = 50,
    start=None,
    end=None,
    min_days: float = 0.9,
    kinds: tuple[str, ...] | None = None,
    min_volume: float = 0.0,
    since_listing: bool = True,
    min_history: int = 252,
) -> pd.DataFrame:
    """Ativos ordenados por liquidez (volume financeiro médio por pregão) no período.

    n: quantos devolver (0 = todos que passarem nos filtros).
    kinds: códigos CODBDI a considerar (ver KIND_CODES); None = todos da base. A categoria de cada
        ativo é a do seu pregão MAIS RECENTE: a B3 já reclassificou papéis no meio da história
        (ex.: em 10/10/2022 todos os BDRs passaram da categoria 02 para 34).
    min_days: fração mínima de pregões negociados (sobre o período de referência abaixo).
    min_volume: volume financeiro médio mínimo por pregão (R$).
    since_listing: True = o período de referência de cada ativo começa na sua estreia (inclui IPOs e
        ETFs lançados depois do início); False = exige negociação ao longo de todo o período.
    min_history: com since_listing, mínimo de pregões desde a estreia (252 ≈ 1 ano).
    """
    years = _years_between(start, end)
    df = load_store(years, ["date", "ticker", "isin", "name", "bdi", "volume"])
    df = to_reais(with_symbols(df))
    if start:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end:
        df = df[df["date"] <= pd.Timestamp(end)]
    if df.empty:
        return pd.DataFrame(columns=UNIVERSE_COLUMNS)
    calendar = np.sort(df["date"].unique())  # pregões do período (qualquer ativo negociou)
    # nome e categoria vêm do pregão mais recente de cada ativo
    agg = df.sort_values("date").groupby("ticker").agg(
        name=("name", "last"), bdi=("bdi", "last"), first=("date", "min"),
        days=("date", "nunique"), total=("volume", "sum"),
    )
    if kinds:
        agg = agg[agg["bdi"].isin(kinds)]
    if since_listing:
        # pregões disponíveis desde a estreia de cada ativo até o fim do período
        agg["sessions"] = len(calendar) - np.searchsorted(calendar, agg["first"].to_numpy(), side="left")
        agg = agg[agg["sessions"] >= min_history]
    else:
        agg["sessions"] = len(calendar)
    # média sobre os pregões de referência (não só os negociados): penaliza quem negocia pouco
    agg["avg_volume"] = agg["total"] / agg["sessions"].clip(lower=1)
    agg["kind"] = agg["bdi"].map(kind_of)
    agg = agg[(agg["days"] >= min_days * agg["sessions"]) & (agg["avg_volume"] >= min_volume)]
    agg = agg.sort_values("avg_volume", ascending=False)
    if n:
        agg = agg.head(n)
    return agg.reset_index()[UNIVERSE_COLUMNS]


def load_ohlcv_many(tickers: Iterable[str], start=None, end=None, adjust: bool = True,
                    total_return: bool = False) -> dict[str, pd.DataFrame]:
    """OHLCV de vários ativos lendo a base uma vez só (mais rápido que load_ticker em laço)."""
    tickers = list(dict.fromkeys(t.upper().strip() for t in tickers if t.strip()))
    years = _years_between(start, end)
    df = load_store(years, ["date", "ticker", "isin", "open", "high", "low", "close", "volume", "quantity"])
    df = with_symbols(df)
    df = df[df["ticker"].isin(tickers)]
    out: dict[str, pd.DataFrame] = {}
    for t, g in df.groupby("ticker", sort=False):
        g = _prepare(g, adjust, t, total_return)
        if start:
            g = g[g.index >= pd.Timestamp(start)]
        if end:
            g = g[g.index <= pd.Timestamp(end)]
        if len(g):
            x = g[["open", "high", "low", "close", "volume"]].astype(float)
            x.index.name = "date"
            out[t] = x
    return out
