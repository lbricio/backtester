"""Otimização de parâmetros por busca em grade."""

from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
import pandas as pd

from quantbt.engine import BacktestConfig, run_backtest
from quantbt.strategies import Strategy

MAX_COMBINATIONS = 20_000


def frange(lo: float, hi: float, step: float, as_int: bool) -> list:
    if step <= 0 or hi < lo:
        return [int(lo) if as_int else lo]
    n = int(np.floor((hi - lo) / step + 1e-9)) + 1
    vals = [lo + i * step for i in range(n)]
    return [int(round(v)) for v in vals] if as_int else [round(v, 10) for v in vals]


def count_combinations(grid: dict[str, list]) -> int:
    return math.prod(len(v) for v in grid.values()) if grid else 0


def grid_search(
    df: pd.DataFrame,
    strategy: Strategy,
    grid: dict[str, list],
    config: BacktestConfig,
    fixed: dict | None = None,
    progress: Callable[[int, int], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    keys = list(grid)
    total = count_combinations(grid)
    # checa o limite ANTES de gerar as combinações (uma grade enorme esgotaria a memória)
    if total > MAX_COMBINATIONS:
        raise ValueError(f"{total:,} combinações — o limite é {MAX_COMBINATIONS:,}. Aumente os passos.")
    rows = []
    for i, combo in enumerate(itertools.product(*(grid[k] for k in keys)), 1):
        if should_stop and should_stop():
            break
        params = {**(fixed or {}), **dict(zip(keys, combo))}
        try:
            sig = strategy.generate_signals(df, **params)
            res = run_backtest(df, sig, config, params)
            rows.append({**dict(zip(keys, combo)), **res.metrics})
        except Exception as e:  # combinação inválida não derruba a otimização
            rows.append({**dict(zip(keys, combo)), "erro": str(e)})
        if progress:
            progress(i, total)
    return pd.DataFrame(rows)
