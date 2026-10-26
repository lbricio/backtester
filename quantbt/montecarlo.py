"""Simulação de Monte Carlo do preço futuro de um ativo.

Modelos (todos trabalham com retornos logarítmicos diários):
- gbm:       movimento browniano geométrico (choques normais).
- t:         como o gbm, mas com choques t-Student (4 graus de liberdade), para caudas gordas.
- bootstrap: sorteia retornos históricos reais, com reposição.
- block:     sorteia blocos de dias consecutivos (mantém sequências de alta/baixa volatilidade).

Tendência (drift), aplicada a todos os modelos:
- historical: a média histórica do período de estimação (pode extrapolar tendências recentes).
- risk_free:  rende em média a taxa livre de risco (mundo "neutro ao risco").
- zero:       preço esperado igual ao atual (sem tendência).
- custom:     retorno esperado anual informado pelo usuário.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from quantbt.metrics import periods_per_year

MODELS = {
    "Normal (GBM)": "gbm",
    "Caudas gordas (t-Student)": "t",
    "Histórico (bootstrap)": "bootstrap",
    "Histórico em blocos": "block",
}
DRIFTS = {
    "Histórica": "historical",
    "Taxa livre de risco": "risk_free",
    "Sem tendência": "zero",
    "Personalizada": "custom",
}
MODEL_INFO = {
    "gbm": "Choques normais com a volatilidade histórica. É o modelo clássico, mas subestima quedas extremas.",
    "t": "Choques t-Student (4 graus de liberdade): mesma volatilidade, mas dias extremos são mais frequentes.",
    "bootstrap": "Sorteia retornos diários que aconteceram de fato, preservando caudas e assimetria reais.",
    "block": "Sorteia blocos de dias consecutivos: preserva também os períodos de calmaria e de turbulência.",
}
MAX_PATHS = 50_000
T_DF = 4.0


@dataclass
class MCConfig:
    model: str = "gbm"
    horizon: int = 252  # pregões à frente
    n_paths: int = 5_000
    drift: str = "historical"
    custom_return: float = 0.10  # retorno anual esperado (drift "custom")
    risk_free: float = 0.0  # taxa anual (drift "risk_free")
    vol_override: float = 0.0  # volatilidade anual; 0 = histórica
    block: int = 20  # tamanho do bloco (modelo "block")
    seed: int | None = None


@dataclass
class Estimate:
    """Parâmetros estimados do histórico."""

    s0: float
    last_date: pd.Timestamp
    returns: np.ndarray  # retornos log diários
    ppy: float
    mu_d: float  # média diária dos retornos log
    sigma_d: float  # desvio diário

    @property
    def mu_annual(self) -> float:
        """Retorno anual esperado implícito (aritmético)."""
        return float(np.exp((self.mu_d + self.sigma_d**2 / 2) * self.ppy) - 1)

    @property
    def sigma_annual(self) -> float:
        return float(self.sigma_d * np.sqrt(self.ppy))


def estimate(prices: pd.Series, years: float | None = 2.0) -> Estimate:
    """Estima volatilidade e tendência a partir dos últimos `years` anos de fechamentos."""
    p = prices.dropna().astype(float)
    p = p[p > 0]
    if years:
        p = p[p.index >= p.index[-1] - pd.DateOffset(days=int(365.25 * years))]
    r = np.log(p / p.shift(1)).dropna().to_numpy()
    r = r[np.isfinite(r)]
    if len(r) < 20:
        raise ValueError("Histórico curto demais para estimar a volatilidade (mínimo ~20 pregões).")
    return Estimate(s0=float(p.iloc[-1]), last_date=p.index[-1], returns=r, ppy=periods_per_year(p.index),
                    mu_d=float(r.mean()), sigma_d=float(r.std(ddof=1)))


def daily_params(est: Estimate, cfg: MCConfig) -> tuple[float, float]:
    """(média diária do retorno log, desvio diário) usados na simulação."""
    sigma = cfg.vol_override / np.sqrt(est.ppy) if cfg.vol_override > 0 else est.sigma_d
    if cfg.drift == "historical":
        mu = est.mu_d
    elif cfg.drift == "risk_free":
        mu = np.log1p(cfg.risk_free) / est.ppy - sigma**2 / 2
    elif cfg.drift == "zero":
        mu = -(sigma**2) / 2  # preço esperado = preço atual
    elif cfg.drift == "custom":
        mu = np.log1p(cfg.custom_return) / est.ppy - sigma**2 / 2
    else:
        raise ValueError(f"Tendência desconhecida: {cfg.drift}")
    return float(mu), float(sigma)


def _shocks(est: Estimate, cfg: MCConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """Matriz (n, horizon) de retornos log diários padronizados ou sorteados."""
    h = cfg.horizon
    mu, sigma = daily_params(est, cfg)
    if cfg.model == "gbm":
        z = rng.standard_normal((n, h), dtype=np.float32)
    elif cfg.model == "t":
        z = (rng.standard_t(T_DF, (n, h)) / np.sqrt(T_DF / (T_DF - 2))).astype(np.float32)  # variância 1
    elif cfg.model in ("bootstrap", "block"):
        hist = ((est.returns - est.mu_d) / est.sigma_d).astype(np.float32)  # retornos reais padronizados
        if cfg.model == "bootstrap":
            z = hist[rng.integers(0, len(hist), (n, h))]
        else:
            b = int(np.clip(cfg.block, 2, max(2, len(hist) // 2)))
            nb = -(-h // b)
            starts = rng.integers(0, len(hist) - b + 1, (n, nb))
            idx = (starts[:, :, None] + np.arange(b)[None, None, :]).reshape(n, nb * b)[:, :h]
            z = hist[idx]
    else:
        raise ValueError(f"Modelo desconhecido: {cfg.model}")
    return mu + sigma * z


def simulate(est: Estimate, cfg: MCConfig, n: int | None = None, rng: np.random.Generator | None = None) -> np.ndarray:
    """Trajetórias de preço: matriz (n, horizon + 1), começando no preço atual."""
    n = int(min(n or cfg.n_paths, MAX_PATHS))
    rng = rng or np.random.default_rng(cfg.seed)
    r = _shocks(est, cfg, n, rng)
    paths = np.empty((n, cfg.horizon + 1), dtype=np.float32)
    paths[:, 0] = est.s0
    paths[:, 1:] = est.s0 * np.exp(np.cumsum(r, axis=1, dtype=np.float64))
    return paths


def simulate_batches(est: Estimate, cfg: MCConfig, batches: int = 40):
    """Gera as trajetórias em lotes (para a animação): devolve um gerador de matrizes."""
    rng = np.random.default_rng(cfg.seed)
    total = int(min(cfg.n_paths, MAX_PATHS))
    sizes = np.diff(np.linspace(0, total, max(1, min(batches, total)) + 1).astype(int))
    for size in sizes:
        if size:
            yield simulate(est, cfg, int(size), rng)


PCTS = (5, 25, 50, 75, 95)


@dataclass
class MCStats:
    n: int
    s0: float
    pcts: dict[int, np.ndarray]  # percentil -> curva ao longo do tempo
    final: np.ndarray
    mean_final: float
    prob_up: float
    var95: float  # perda no pior 5% (retorno, negativo)
    cvar95: float  # perda média dos piores 5%
    median_max_dd: float
    target: float | None = None
    prob_above_target: float | None = None
    prob_touch_target: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def median_final(self) -> float:
        return float(self.pcts[50][-1])


def stats(paths: np.ndarray, s0: float, target: float | None = None) -> MCStats:
    final = paths[:, -1].astype(np.float64)
    ret = final / s0 - 1
    q = np.percentile(paths, PCTS, axis=0)
    var = float(np.percentile(ret, 5))
    tail = ret[ret <= var]
    peak = np.maximum.accumulate(paths, axis=1)
    max_dd = (paths / peak - 1).min(axis=1)
    out = MCStats(
        n=len(paths), s0=s0, pcts={p: q[i] for i, p in enumerate(PCTS)}, final=final,
        mean_final=float(final.mean()), prob_up=float((final > s0).mean()),
        var95=var, cvar95=float(tail.mean()) if len(tail) else var,
        median_max_dd=float(np.median(max_dd)),
    )
    if target and target > 0:
        out.target = target
        out.prob_above_target = float((final >= target).mean()) if target >= s0 else float((final <= target).mean())
        touched = paths.max(axis=1) >= target if target >= s0 else paths.min(axis=1) <= target
        out.prob_touch_target = float(touched.mean())
    return out


def future_dates(last: pd.Timestamp, horizon: int, ppy: float) -> pd.DatetimeIndex:
    """Datas aproximadas dos pregões simulados (dias úteis; corridos se o ativo negocia 7 dias)."""
    freq = "D" if ppy > 300 else "B"
    return pd.date_range(last, periods=horizon + 1, freq=freq)
