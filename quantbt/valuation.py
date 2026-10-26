"""Preço justo de ações por métodos fundamentalistas.

- Graham:  √(22,5 × LPA × VPA). Supõe P/L ≤ 15 e P/VPA ≤ 1,5 (15 × 1,5 = 22,5).
- Bazin:   preço teto = dividendos por ação ÷ dividend yield mínimo desejado (clássico: 6%).
- Gordon:  P = D₁ / (k − g), com D₁ = D₀ × (1 + g). Para empresas maduras que pagam dividendos.
- FCD:     fluxo de caixa livre por ação, crescendo a g₁ por N anos e depois a g₂ para sempre,
           descontado a k. Versão Monte Carlo sorteia g₁, g₂ e k para mostrar a incerteza.

Os dados fundamentalistas vêm do Yahoo Finance e podem estar desatualizados ou em outra base
(ex.: units somam várias ações). Confira com o RI da empresa antes de decidir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Fundamentals:
    ticker: str = ""
    price: float = 0.0
    eps: float = 0.0  # LPA (lucro por ação, 12 meses)
    bvps: float = 0.0  # VPA (valor patrimonial por ação)
    dps_12m: float = 0.0  # dividendos por ação nos últimos 12 meses
    dps_avg5: float = 0.0  # média anual de dividendos nos últimos 5 anos completos
    fcf_ps: float = 0.0  # fluxo de caixa livre por ação
    source: str = ""
    warnings: list[str] = field(default_factory=list)


def graham(eps: float, bvps: float, multiplier: float = 22.5) -> float | None:
    if eps <= 0 or bvps <= 0:
        return None  # sem lucro ou patrimônio positivo a fórmula não se aplica
    return math.sqrt(multiplier * eps * bvps)


def bazin(dps: float, min_yield: float = 0.06) -> float | None:
    if dps <= 0 or min_yield <= 0:
        return None
    return dps / min_yield


def gordon(d0: float, k: float, g: float) -> float | None:
    if d0 <= 0 or k <= g:
        return None  # crescimento perpétuo não pode igualar/superar o custo de capital
    return d0 * (1 + g) / (k - g)


def dcf(fcf0: float, g1: float, years: int, g2: float, k: float) -> float | None:
    """Valor por ação: FCF crescendo a g1 por `years` anos, perpetuidade a g2, desconto k."""
    if fcf0 <= 0 or k <= g2 or years < 1:
        return None
    t = np.arange(1, years + 1)
    flows = fcf0 * (1 + g1) ** t
    pv = float((flows / (1 + k) ** t).sum())
    terminal = flows[-1] * (1 + g2) / (k - g2)
    return pv + terminal / (1 + k) ** years


def dcf_monte_carlo(fcf0: float, g1: float, years: int, g2: float, k: float,
                    sd_g1: float, sd_g2: float, sd_k: float, n: int = 20_000,
                    seed: int | None = None) -> np.ndarray:
    """Distribuição do FCD sorteando g1, g2 e k (normais, em fração ao ano).

    Sorteios em que k ≤ g2 (perpetuidade infinita) são descartados.
    """
    if fcf0 <= 0 or years < 1:
        return np.empty(0)
    rng = np.random.default_rng(seed)
    G1 = rng.normal(g1, sd_g1, n)
    G2 = rng.normal(g2, sd_g2, n)
    K = rng.normal(k, sd_k, n)
    ok = (K - G2 > 0.005) & (K > -0.5) & (G1 > -0.9)  # margem mínima entre k e g2
    G1, G2, K = G1[ok], G2[ok], K[ok]
    t = np.arange(1, years + 1)[None, :]
    flows = fcf0 * (1 + G1[:, None]) ** t
    pv = (flows / (1 + K[:, None]) ** t).sum(axis=1)
    terminal = flows[:, -1] * (1 + G2) / (K - G2)
    return pv + terminal / (1 + K) ** years


def margin(fair: float | None, price: float) -> float | None:
    """Margem de segurança: quanto o preço justo está acima (+) ou abaixo (−) do preço atual."""
    if fair is None or price <= 0:
        return None
    return fair / price - 1


def dividends_per_share(divs: pd.Series, today: pd.Timestamp | None = None) -> tuple[float, float]:
    """(soma dos últimos 12 meses, média anual dos últimos 5 anos completos)."""
    if divs is None or len(divs) == 0:
        return 0.0, 0.0
    d = divs.copy()
    if getattr(d.index, "tz", None) is not None:
        d.index = d.index.tz_localize(None)
    today = today or pd.Timestamp.today().normalize()
    last12 = float(d[d.index > today - pd.Timedelta(days=365)].sum())
    year = today.year
    yearly = d.groupby(d.index.year).sum()
    full = [float(yearly.get(y, 0.0)) for y in range(year - 5, year)]
    return last12, float(np.mean(full)) if full else 0.0


def fetch_fundamentals(ticker: str) -> Fundamentals:
    """Busca LPA, VPA, dividendos e FCF no Yahoo Finance (códigos da B3 ganham '.SA')."""
    import warnings

    import yfinance as yf

    from quantbt.data import _B3_LIKE

    t = ticker.upper().strip()
    sym = f"{t}.SA" if _B3_LIKE.match(t) else t
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tk = yf.Ticker(sym)
        info = tk.info or {}
        divs = tk.dividends
    if not info or (info.get("currentPrice") is None and info.get("regularMarketPrice") is None):
        raise ValueError(f"Yahoo não retornou dados fundamentalistas para {sym}.")
    f = Fundamentals(ticker=t, source=f"Yahoo Finance ({sym})")
    f.price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
    f.eps = float(info.get("trailingEps") or 0)
    f.bvps = float(info.get("bookValue") or 0)
    f.dps_12m, f.dps_avg5 = dividends_per_share(divs)
    fcf, shares = info.get("freeCashflow"), info.get("sharesOutstanding")
    f.fcf_ps = float(fcf) / float(shares) if fcf and shares else 0.0
    if not f.fcf_ps:
        f.warnings.append("Sem fluxo de caixa livre no Yahoo (comum em bancos): o FCD fica de fora.")
    if t.endswith("11"):
        f.warnings.append("Units/FIIs: o Yahoo pode misturar valores por ação e por unit. Confira LPA e VPA.")
    if info.get("financialCurrency") and info.get("financialCurrency") != info.get("currency"):
        f.warnings.append(f"Balanço em {info.get('financialCurrency')} e cotação em {info.get('currency')}.")
    return f


@dataclass
class ValuationParams:
    graham_mult: float = 22.5
    bazin_yield: float = 0.06
    bazin_use_avg: bool = True  # usa a média de 5 anos (mais estável que só os últimos 12 meses)
    k: float = 0.13  # custo de capital (taxa de desconto) anual
    g_perp: float = 0.04  # crescimento perpétuo (Gordon e perpetuidade do FCD)
    g_years: float = 0.08  # crescimento do FCF nos primeiros anos
    years: int = 5


def evaluate(f: Fundamentals, p: ValuationParams) -> pd.DataFrame:
    """Tabela com o preço justo de cada método e a margem em relação ao preço atual."""
    dps = f.dps_avg5 if (p.bazin_use_avg and f.dps_avg5 > 0) else f.dps_12m
    rows = [
        ("Graham", graham(f.eps, f.bvps, p.graham_mult),
         f"√({p.graham_mult:g} × LPA {f.eps:.2f} × VPA {f.bvps:.2f})", "Precisa de lucro e patrimônio positivos"),
        ("Bazin", bazin(dps, p.bazin_yield),
         f"Dividendos {dps:.2f} ÷ {p.bazin_yield:.1%}", "Preço teto para quem busca dividendos"),
        ("Gordon", gordon(dps, p.k, p.g_perp),
         f"D₀ {dps:.2f} × (1 + {p.g_perp:.1%}) ÷ ({p.k:.1%} − {p.g_perp:.1%})", "Empresas maduras pagadoras de dividendos"),
        ("Fluxo de caixa descontado", dcf(f.fcf_ps, p.g_years, p.years, p.g_perp, p.k),
         f"FCF {f.fcf_ps:.2f}, {p.g_years:.0%} por {p.years} anos, perpetuidade {p.g_perp:.1%}, desconto {p.k:.1%}",
         "Mais sensível às premissas de crescimento e desconto"),
    ]
    out = pd.DataFrame(rows, columns=["metodo", "preco_justo", "conta", "obs"])
    out["margem"] = [margin(v, f.price) for v in out["preco_justo"]]
    return out
