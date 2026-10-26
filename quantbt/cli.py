"""Linha de comando: base B3 e correlação do universo inteiro sem abrir a janela.

Exemplos:
    uv run quantbt-cli b3 --update                     # baixa o ano corrente
    uv run quantbt-cli b3 --from 1986 --to 2026        # baixa um intervalo de anos
    uv run quantbt-cli corr --start 2023-01-01 --min-volume 1000000 --out resultados/
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

from quantbt import b3, external
from quantbt.correlation import FREQS, METHODS, MethodParams, analyze

PRESET_ARGS = {"moedas": "Dólar, mini dólar e moedas", "commodities": "Commodities", "cripto": "Bitcoin e cripto",
               "indices": "Índices e mini índice", "tudo": "Tudo (macro + commodities + cripto)"}
KIND_ARGS = {"acoes": "Ações / units", "bdrs": "BDRs", "fiis": "FIIs / Fiagro", "etfs": "ETFs / fundos"}


def cmd_b3(args) -> int:
    this = date.today().year
    if args.update:
        years = [this]
    else:
        years = list(range(args.to_year or this, (args.from_year or this) - 1, -1))
        if args.missing:
            have = set(b3.available_years())
            years = [y for y in years if y not in have or y == this]

    def job(y):
        for _ in range(3):
            try:
                return y, b3.download_year(y), None
            except Exception as e:  # noqa: PERF203
                err = e
        return y, 0, err

    print(f"Baixando {len(years)} ano(s) para {b3.store_dir()}")
    with ThreadPoolExecutor(args.jobs) as ex:
        for y, n, err in ex.map(job, years):
            print(f"  {y}: " + (f"{n:,} registros" if err is None else f"FALHOU ({err})"), flush=True)
    print("Anos na base:", b3.available_years())
    return 0


def cmd_corr(args) -> int:
    t0 = time.time()
    kinds = None
    if args.kind:
        kinds = tuple(c for k in args.kind for c in b3.KIND_CODES[KIND_ARGS[k]])
    uni = b3.liquid_universe(args.max_assets, args.start, args.end, args.min_days, kinds, args.min_volume,
                             since_listing=not args.full_period, min_history=int(args.min_history * 252))
    if len(uni) < 2:
        print("Menos de 2 ativos passaram nos filtros.", file=sys.stderr)
        return 1
    extra = [c.upper() for c in (args.extra or [])]
    for p in args.preset or []:
        extra += external.PRESETS[PRESET_ARGS[p]]
    extra = [c for c in dict.fromkeys(extra) if c not in set(uni["ticker"])]
    bad = [c for c in extra if not external.is_external(c)]
    if bad:
        print(f"Não estão no catálogo de externos: {', '.join(bad)} (veja: quantbt-cli externos --list)")
        extra = [c for c in extra if c not in bad]
    if extra:
        ext_rows = [{"ticker": c, "name": external.describe(c), "kind": external.CATALOG[c].group} for c in extra]
        uni = pd.concat([uni, pd.DataFrame(ext_rows)], ignore_index=True)
    print(f"Universo: {len(uni)} ativos — " + ", ".join(f"{k}: {v}" for k, v in uni["kind"].value_counts().items()))
    prices = external.combined_closes(
        uni["ticker"], lambda ts: b3.load_closes(ts, args.start, args.end, adjust=not args.no_adjust),
        args.start, args.end, in_brl=args.brl)
    params = MethodParams(halflife=args.halflife, tail_q=args.tail, neutral=args.neutral / 100,
                          market_name=args.market.upper())
    if args.market:
        mk = args.market.upper()
        try:
            params.market = prices[mk] if mk in prices else external.combined_closes(
                [mk], lambda ts: b3.load_closes(ts, args.start, args.end), args.start, args.end)[mk]
        except Exception:
            print(f"Fator de mercado {mk} não encontrado: usando a média do universo.")
    compare = [m for m in METHODS.values() if m != args.method] if args.compare else []
    res = analyze(prices, freq=FREQS[args.freq], method=args.method, min_obs=args.min_obs,
                  rolling_window=args.window, min_coverage=args.min_coverage, rolling_top=args.rolling_top,
                  compare=compare, params=params)
    for note in res.notes:
        print(note)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = res.pairs.merge(uni[["ticker", "name", "kind"]].add_prefix("a_"), left_on="a", right_on="a_ticker", how="left")
    pairs = pairs.merge(uni[["ticker", "name", "kind"]].add_prefix("b_"), left_on="b", right_on="b_ticker", how="left")
    pairs = pairs.drop(columns=["a_ticker", "b_ticker"])
    pairs.to_csv(out / "ranking_pares.csv", index=False)
    res.matrix.round(4).to_csv(out / "matriz_correlacao.csv")
    uni.to_csv(out / "universo.csv", index=False)

    view = pairs[~pairs["same_issuer"]] if not args.keep_same else pairs
    print(f"{len(res.pairs):,} pares calculados em {time.time() - t0:,.1f}s "
          f"({len(res.returns):,} períodos, {res.matrix.shape[0]} ativos; descartados por cobertura: {len(res.dropped)})")
    if args.show:
        print(f"\nTop {args.show} pares mais correlacionados" + ("" if args.keep_same else " (sem mesma empresa)") + ":")
    for r in view.head(args.show).itertuples():
        extra = f"  ({(1 + r.corr) / 2:.0%} dos dias juntos)" if args.method == "sign" else ""
        print(f"  {r.a:>8} × {r.b:<8} {r.corr:+.3f}{extra}   obs {r.obs:>5}   {r.a_name} / {r.b_name}")
    print(f"\nArquivos em {out.resolve()}/")
    return 0


def cmd_externos(args) -> int:
    if args.list or not args.update:
        cat = external.catalog_table()
        for grp, sub in cat.groupby("grupo", sort=False):
            print(f"\n{grp}")
            for r in sub.itertuples():
                print(f"  {r.apelido:<9} {r.nome:<42} {r.fonte}:{r.código}" + (f"  — {r.obs}" if r.obs else ""))
    if args.update:
        print(f"Atualizando cache em {external.cache_dir()}")
        external.update_all(log=lambda m: print("  " + m, flush=True))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="quantbt-cli", description="quantbt pela linha de comando")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("b3", help="baixa séries históricas da B3 (COTAHIST)")
    p.add_argument("--from", dest="from_year", type=int, help="primeiro ano (padrão: ano corrente)")
    p.add_argument("--to", dest="to_year", type=int, help="último ano (padrão: ano corrente)")
    p.add_argument("--update", action="store_true", help="só o ano corrente")
    p.add_argument("--missing", action="store_true", help="só anos que faltam (e o corrente)")
    p.add_argument("--jobs", type=int, default=3, help="downloads simultâneos")
    p.set_defaults(fn=cmd_b3)

    p = sub.add_parser("corr", help="correlação do universo da base B3")
    p.add_argument("--start", default=str(date.today().replace(year=date.today().year - 10)), help="início (padrão: 10 anos atrás)")
    p.add_argument("--end", default=None)
    p.add_argument("--kind", nargs="*", choices=list(KIND_ARGS), help="tipos de ativo (padrão: todos)")
    p.add_argument("--min-volume", type=float, default=500_000, help="volume médio mínimo por pregão (R$)")
    p.add_argument("--min-days", type=float, default=0.9, help="fração mínima de pregões negociados")
    p.add_argument("--max-assets", type=int, default=0, help="limita aos N mais líquidos (0 = todos)")
    p.add_argument("--full-period", action="store_true",
                   help="exige negociação no período inteiro (padrão: conta desde a estreia de cada ativo)")
    p.add_argument("--min-history", type=float, default=1.0, help="histórico mínimo desde a estreia (anos)")
    p.add_argument("--freq", choices=list(FREQS), default="Diária")
    p.add_argument("--method", choices=list(METHODS.values()), default="pearson",
                   help="sign = concordância de direção; ewma; residual (sem mercado); stress (piores dias)")
    p.add_argument("--compare", action="store_true", help="calcula também os outros métodos (colunas corr_*)")
    p.add_argument("--market", default="BOVA11", help="fator de mercado (vazio = média do universo)")
    p.add_argument("--halflife", type=float, default=126, help="meia-vida do EWMA (períodos)")
    p.add_argument("--tail", type=float, default=0.2, help="estresse: fração dos piores dias do mercado")
    p.add_argument("--neutral", type=float, default=0.1, help="direção: zona neutra em %% (ex.: 0.1)")
    p.add_argument("--min-obs", type=int, default=60)
    p.add_argument("--min-coverage", type=float, default=0.5)
    p.add_argument("--window", type=int, default=60, help="janela da correlação móvel (0 desliga)")
    p.add_argument("--rolling-top", type=int, default=3000, help="pares com correlação móvel calculada")
    p.add_argument("--no-adjust", action="store_true", help="não corrigir desdobramentos")
    p.add_argument("--keep-same", action="store_true", help="mostrar pares da mesma empresa no resumo")
    p.add_argument("--show", type=int, default=25)
    p.add_argument("--out", default="resultados_correlacao")
    p.add_argument("--extra", nargs="*", help="ativos externos, ex.: DOLAR WDO BTC OURO PETROLEO")
    p.add_argument("--preset", nargs="*", choices=list(PRESET_ARGS), help="grupos de externos")
    p.add_argument("--brl", action="store_true", help="converte externos cotados em US$ para R$")
    p.set_defaults(fn=cmd_corr)

    p = sub.add_parser("externos", help="catálogo de dólar, commodities, cripto e índices")
    p.add_argument("--list", action="store_true", help="lista o catálogo")
    p.add_argument("--update", action="store_true", help="baixa/atualiza todos (10 anos)")
    p.set_defaults(fn=cmd_externos)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"Erro: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
