"""Preço justo de opções europeias: Black-Scholes e Monte Carlo.

Convenção da B3: prazo em dias úteis, ano de 252 dias úteis. Taxas em % ao ano, capitalizadas
continuamente nas fórmulas (r_c = ln(1 + r)).

Observação: na B3, opções de compra sobre ações costumam ser americanas. Sem dividendos até o
vencimento, a call americana vale o mesmo que a europeia; a put americana pode valer um pouco mais.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DAYS_YEAR = 252


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _npdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@dataclass
class OptionSpec:
    kind: str  # "call" ou "put"
    s: float  # preço do ativo
    k: float  # strike
    days: int  # dias úteis até o vencimento
    r: float  # taxa livre de risco anual (ex.: 0.105)
    sigma: float  # volatilidade anual (ex.: 0.30)
    q: float = 0.0  # dividend yield anual

    @property
    def t(self) -> float:
        return max(self.days, 0) / DAYS_YEAR

    @property
    def rc(self) -> float:
        return math.log1p(self.r)

    @property
    def qc(self) -> float:
        return math.log1p(self.q)


def black_scholes(o: OptionSpec) -> float:
    t, s, k = o.t, o.s, o.k
    if t <= 0 or o.sigma <= 0:
        intrinsic = max(s - k, 0.0) if o.kind == "call" else max(k - s, 0.0)
        return intrinsic * (math.exp(-o.rc * t) if t > 0 else 1.0)
    d1 = (math.log(s / k) + (o.rc - o.qc + o.sigma**2 / 2) * t) / (o.sigma * math.sqrt(t))
    d2 = d1 - o.sigma * math.sqrt(t)
    df_r, df_q = math.exp(-o.rc * t), math.exp(-o.qc * t)
    if o.kind == "call":
        return s * df_q * _ncdf(d1) - k * df_r * _ncdf(d2)
    return k * df_r * _ncdf(-d2) - s * df_q * _ncdf(-d1)


def greeks(o: OptionSpec) -> dict[str, float]:
    """Gregas de Black-Scholes. Vega e rho por 1 ponto percentual; theta por dia útil."""
    t = o.t
    if t <= 0 or o.sigma <= 0:
        itm = (o.s > o.k) if o.kind == "call" else (o.s < o.k)
        return {"delta": (1.0 if o.kind == "call" else -1.0) if itm else 0.0, "gamma": 0.0, "vega": 0.0,
                "theta": 0.0, "rho": 0.0, "prob_exercicio": float(itm)}
    sq = math.sqrt(t)
    d1 = (math.log(o.s / o.k) + (o.rc - o.qc + o.sigma**2 / 2) * t) / (o.sigma * sq)
    d2 = d1 - o.sigma * sq
    df_r, df_q = math.exp(-o.rc * t), math.exp(-o.qc * t)
    gamma = df_q * _npdf(d1) / (o.s * o.sigma * sq)
    vega = o.s * df_q * _npdf(d1) * sq / 100
    if o.kind == "call":
        delta = df_q * _ncdf(d1)
        theta = (-o.s * df_q * _npdf(d1) * o.sigma / (2 * sq) - o.rc * o.k * df_r * _ncdf(d2)
                 + o.qc * o.s * df_q * _ncdf(d1))
        rho = o.k * t * df_r * _ncdf(d2) / 100
        prob = _ncdf(d2)
    else:
        delta = -df_q * _ncdf(-d1)
        theta = (-o.s * df_q * _npdf(d1) * o.sigma / (2 * sq) + o.rc * o.k * df_r * _ncdf(-d2)
                 - o.qc * o.s * df_q * _ncdf(-d1))
        rho = -o.k * t * df_r * _ncdf(-d2) / 100
        prob = _ncdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta / DAYS_YEAR, "rho": rho,
            "prob_exercicio": prob}


def implied_vol(o: OptionSpec, market_price: float, lo: float = 1e-4, hi: float = 5.0) -> float | None:
    """Volatilidade implícita (bisseção). None se o preço estiver fora do intervalo possível."""
    def f(sig: float) -> float:
        return black_scholes(OptionSpec(o.kind, o.s, o.k, o.days, o.r, sig, o.q)) - market_price

    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:
        return None
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-7:
            break
    return (lo + hi) / 2


@dataclass
class MCOptionResult:
    n: int
    price: float
    stderr: float
    payoffs_sample: np.ndarray  # amostra de preços finais (para o histograma)
    history: list[tuple[int, float, float]]  # (n acumulado, preço, erro-padrão) — curva de convergência

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.stderr, self.price + 1.96 * self.stderr


class MCOptionPricer:
    """Precificação por Monte Carlo neutro ao risco, em lotes (para a animação de convergência).

    Usa variáveis antitéticas (cada choque z vem com -z), o que reduz o erro com o mesmo custo.
    """

    def __init__(self, o: OptionSpec, seed: int | None = None) -> None:
        self.o = o
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self._sum = 0.0
        self._sum2 = 0.0
        self.history: list[tuple[int, float, float]] = []
        self._sample: list[np.ndarray] = []

    def step(self, n_pairs: int) -> MCOptionResult:
        o = self.o
        t = o.t
        z = self.rng.standard_normal(n_pairs)
        drift = (o.rc - o.qc - o.sigma**2 / 2) * t
        vol = o.sigma * math.sqrt(t)
        st_up = o.s * np.exp(drift + vol * z)
        st_dn = o.s * np.exp(drift - vol * z)
        pay = (np.maximum(st_up - o.k, 0) + np.maximum(st_dn - o.k, 0)) / 2 if o.kind == "call" else \
              (np.maximum(o.k - st_up, 0) + np.maximum(o.k - st_dn, 0)) / 2
        disc = math.exp(-o.rc * t) * pay  # cada par antitético é uma observação
        self.n += n_pairs
        self._sum += float(disc.sum())
        self._sum2 += float((disc**2).sum())
        if sum(len(s) for s in self._sample) < 20_000:
            self._sample.append(st_up[:2000])
        mean, se = self._estimate()
        self.history.append((self.n * 2, mean, se))  # nº de trajetórias = 2 × pares
        return self.result()

    def _estimate(self) -> tuple[float, float]:
        n = max(self.n, 1)
        mean = self._sum / n
        var = max(self._sum2 / n - mean**2, 0.0)
        return mean, math.sqrt(var / n)

    def result(self) -> MCOptionResult:
        mean, se = self._estimate()
        sample = np.concatenate(self._sample) if self._sample else np.empty(0)
        return MCOptionResult(n=self.n * 2, price=mean, stderr=se, payoffs_sample=sample, history=list(self.history))


def mc_price(o: OptionSpec, n_paths: int = 200_000, seed: int | None = None) -> MCOptionResult:
    return MCOptionPricer(o, seed).step(max(1, n_paths // 2))
