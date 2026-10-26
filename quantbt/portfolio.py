"""Backtest de carteira: estratégias que escolhem pesos para vários ativos ao mesmo tempo.

Modelo de execução (igual ao backtest de um ativo):
- A estratégia define pesos-alvo no FECHAMENTO de uma data de rebalanceamento, usando só dados
  até ali; a carteira é ajustada na ABERTURA do pregão seguinte.
- Custos: comissão e slippage sobre o valor girado; o caixa rende o CDI (opcional).
- Pesos negativos = posição vendida (usado em pares). A soma dos |pesos| é limitada a 1.

Universo dinâmico (opcional): a cada rebalanceamento, só são elegíveis os ativos mais líquidos
dos 3 meses anteriores. Evita o viés de sobrevivência de escolher hoje as empresas que deram certo.

Estratégias (as clássicas de gestoras quantitativas):
- Momentum entre ativos (12-1): compra os N que mais subiram em 12 meses, sem o último mês.
- Baixa volatilidade: compra os N menos voláteis (anomalia de baixo risco).
- Paridade de risco: todos os ativos, com peso inverso à volatilidade.
- Igual peso (1/N): referência clássica, rebalanceada.
- Reversão de curto prazo: compra os N que mais caíram na última semana/mês.
- Rotação com filtro de tendência (dual momentum): momentum, mas só ativos acima da média de 200;
  o que não for alocado fica no caixa.
- Pares (arbitragem estatística): compra um e vende o outro quando o spread se afasta da média.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from quantbt.engine import BacktestConfig, cash_returns
from quantbt.metrics import compute_metrics, periods_per_year
from quantbt.strategies import Param, period_end_mask

REBALANCE = {"Semanal": "W-FRI", "Mensal": "M", "Trimestral": "Q"}


@dataclass
class Panels:
    open: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def assets(self) -> list[str]:
        return self.close.columns.tolist()


def make_panels(data: dict[str, pd.DataFrame]) -> Panels:
    """Alinha OHLCV de vários ativos num calendário comum (união das datas)."""
    if not data:
        raise ValueError("Nenhum ativo com dados.")
    close = pd.DataFrame({k: v["close"] for k, v in data.items()}).sort_index()
    open_ = pd.DataFrame({k: v["open"] for k, v in data.items()}).reindex(close.index)
    # na base B3 o volume já é financeiro (R$); externos costumam vir sem volume (0)
    vol = pd.DataFrame({k: v["volume"] for k, v in data.items()}).reindex(close.index)
    return Panels(open=open_, close=close, volume=vol)


# ---------------------------------------------------------------- estratégias de carteira
class PortfolioStrategy:
    name = "Carteira"
    description = ""
    params: list[Param] = []
    rebalance = "M"
    long_short = False

    def defaults(self) -> dict:
        return {p.name: (int(p.default) if p.is_int else p.default) for p in self.params}

    def target_weights(self, close: pd.DataFrame, dates: pd.DatetimeIndex, eligible: pd.DataFrame,
                       **params) -> pd.DataFrame:
        """Pesos-alvo em cada data de `dates` (linhas) usando só preços até a própria data."""
        raise NotImplementedError


def _top(score: pd.DataFrame, eligible: pd.DataFrame, n: int, largest: bool = True) -> pd.DataFrame:
    """Peso igual nos N melhores (ou piores) de cada linha, entre os elegíveis com score válido."""
    s = score.where(eligible & score.notna())
    rank = s.rank(axis=1, ascending=not largest, method="first")
    pick = rank <= n
    count = pick.sum(axis=1).replace(0, np.nan)
    return pick.div(count, axis=0).fillna(0.0)


def _vol(close: pd.DataFrame, n: int) -> pd.DataFrame:
    return np.log(close / close.shift(1)).rolling(int(n), min_periods=int(n * 0.8)).std() * np.sqrt(252)


class XSMomentum(PortfolioStrategy):
    name = "Momentum entre ativos (12-1)"
    description = ("Fator momentum (Jegadeesh e Titman): a cada mês compra os N ativos com maior retorno "
                   "nos últimos 12 meses, ignorando o último mês, com pesos iguais.")
    params = [Param("top", 10, 3, 30, 1, "Quantos ativos"), Param("lookback", 252, 63, 504, 63, "Janela"),
              Param("skip", 21, 0, 63, 21, "Pular (dias)")]

    def target_weights(self, close, dates, eligible, top=10, lookback=252, skip=21):
        mom = close.shift(int(skip)) / close.shift(int(lookback)) - 1
        return _top(mom.reindex(dates), eligible.reindex(dates).fillna(False), int(top))


class LowVolatility(PortfolioStrategy):
    name = "Baixa volatilidade"
    description = ("Anomalia de baixo risco: ações menos voláteis tenderam a render tanto quanto as mais "
                   "arriscadas, com menos queda. Compra os N de menor volatilidade em 1 ano.")
    params = [Param("top", 10, 3, 30, 1, "Quantos ativos"), Param("window", 252, 63, 504, 63, "Janela")]

    def target_weights(self, close, dates, eligible, top=10, window=252):
        vol = _vol(close, window)
        return _top(vol.reindex(dates), eligible.reindex(dates).fillna(False), int(top), largest=False)


class RiskParity(PortfolioStrategy):
    name = "Paridade de risco (inverso da volatilidade)"
    description = ("Todos os ativos elegíveis, cada um com peso proporcional a 1/volatilidade: os mais "
                   "arriscados pesam menos, para cada um contribuir parecido com o risco da carteira.")
    params = [Param("window", 63, 21, 252, 21, "Janela da volatilidade")]

    def target_weights(self, close, dates, eligible, window=63):
        inv = (1 / _vol(close, window)).reindex(dates).where(eligible.reindex(dates).fillna(False))
        return inv.div(inv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


class EqualWeight(PortfolioStrategy):
    name = "Igual peso (1/N)"
    description = "Divide o capital igualmente entre todos os ativos elegíveis e rebalanceia. Referência difícil de bater."
    params = []

    def target_weights(self, close, dates, eligible):
        ok = eligible.reindex(dates).fillna(False) & close.reindex(dates).notna()
        return ok.div(ok.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


class ShortTermReversal(PortfolioStrategy):
    name = "Reversão de curto prazo"
    description = ("Compra os N ativos que mais caíram no período recente (ex.: 1 mês), apostando em "
                   "correção de exageros. Fator clássico de curto prazo; gira bastante.")
    params = [Param("top", 10, 3, 30, 1, "Quantos ativos"), Param("lookback", 21, 5, 63, 1, "Janela (dias)")]

    def target_weights(self, close, dates, eligible, top=10, lookback=21):
        ret = close / close.shift(int(lookback)) - 1
        return _top(ret.reindex(dates), eligible.reindex(dates).fillna(False), int(top), largest=False)


class DualMomentum(PortfolioStrategy):
    name = "Rotação com filtro de tendência (dual momentum)"
    description = ("Momentum relativo + absoluto (Antonacci): escolhe os N de maior momentum, mas só os que "
                   "estão acima da média de 200 dias. As vagas não preenchidas ficam no caixa (CDI).")
    params = [Param("top", 5, 1, 20, 1, "Quantos ativos"), Param("lookback", 252, 63, 504, 63, "Janela"),
              Param("trend", 200, 100, 300, 50, "Média de tendência")]

    def target_weights(self, close, dates, eligible, top=5, lookback=252, trend=200):
        mom = (close / close.shift(int(lookback)) - 1).reindex(dates)
        above = (close > close.rolling(int(trend), min_periods=int(trend)).mean()).reindex(dates)
        ok = eligible.reindex(dates).fillna(False) & above.fillna(False) & (mom > 0)
        s = mom.where(ok)
        pick = s.rank(axis=1, ascending=False, method="first") <= int(top)
        return pick.astype(float) / int(top)  # peso fixo por vaga: vaga vazia = caixa


class PairsTrading(PortfolioStrategy):
    name = "Pares (arbitragem estatística)"
    description = ("Usa os DOIS primeiros ativos da lista. Estima a relação entre eles (regressão dos "
                   "log-preços numa janela móvel) e, quando o spread se afasta Z desvios da média, compra o "
                   "barato e vende o caro; sai quando o spread volta à média. Neutro ao mercado.")
    params = [Param("window", 126, 42, 252, 21, "Janela (dias)"), Param("entry_z", 2.0, 1.0, 3.0, 0.25, "Entrada (z)"),
              Param("exit_z", 0.5, 0.0, 1.5, 0.25, "Saída (z)"), Param("stop_z", 4.0, 2.5, 6.0, 0.5, "Stop (z)")]
    rebalance = "D"
    long_short = True

    def target_weights(self, close, dates, eligible, window=126, entry_z=2.0, exit_z=0.5, stop_z=4.0):
        if close.shape[1] < 2:
            raise ValueError("Pares precisa de 2 ativos (os dois primeiros da lista).")
        a, b = close.columns[:2]
        la, lb = np.log(close[a]), np.log(close[b])
        w = int(window)
        cov = la.rolling(w).cov(lb)
        beta = cov / lb.rolling(w).var()
        alpha = la.rolling(w).mean() - beta * lb.rolling(w).mean()
        spread = la - (alpha + beta * lb)
        z = (spread / spread.rolling(w).std()).to_numpy()
        bt = beta.to_numpy()
        out = np.zeros((len(close), close.shape[1]))
        state, hedge = 0, 0.0
        for i in range(len(close)):
            zi = z[i]
            if not np.isfinite(zi) or not np.isfinite(bt[i]):
                state = 0
            elif state == 0:
                if zi > entry_z or zi < -entry_z:
                    state = -1 if zi > 0 else 1  # z alto: A caro -> vende A, compra B
                    hedge = bt[i]  # razão fixada na entrada
            elif abs(zi) < exit_z or abs(zi) > stop_z or (state == 1 and zi > 0) or (state == -1 and zi < 0):
                state = 0
            if state:
                gross = 1 + abs(hedge)
                out[i, 0], out[i, 1] = state / gross, -state * hedge / gross
        return pd.DataFrame(out, index=close.index, columns=close.columns).reindex(dates).fillna(0.0)


PORTFOLIO_STRATEGIES: list[PortfolioStrategy] = [
    XSMomentum(), LowVolatility(), RiskParity(), EqualWeight(), ShortTermReversal(), DualMomentum(), PairsTrading(),
]


# ---------------------------------------------------------------- motor
@dataclass
class PortfolioResult:
    equity: pd.Series
    weights: pd.DataFrame  # pesos efetivos no fechamento de cada dia
    targets: pd.DataFrame  # pesos-alvo em cada rebalanceamento
    turnover: pd.Series  # giro (soma |Δpeso| / 2) em cada rebalanceamento
    costs: float
    metrics: dict
    benchmarks: dict[str, pd.Series] = field(default_factory=dict)
    benchmark_metrics: dict[str, dict] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def liquidity_mask(p: Panels, top: int, window: int = 63) -> pd.DataFrame:
    """Elegíveis em cada data: os `top` mais líquidos (volume financeiro médio) dos `window` pregões anteriores."""
    vol = p.volume.fillna(0.0).rolling(window, min_periods=int(window * 0.8)).mean()
    rank = vol.rank(axis=1, ascending=False, method="first")
    return (rank <= top) & vol.gt(0)


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "D":
        return index
    return index[period_end_mask(index, freq).to_numpy()]


def run_portfolio(
    panels: Panels,
    strategy: PortfolioStrategy,
    config: BacktestConfig,
    params: dict | None = None,
    rebalance: str | None = None,
    dynamic_top: int = 0,
    warmup: int = 252,
    progress: Callable[[int, int], None] | None = None,
) -> PortfolioResult:
    params = params if params is not None else strategy.defaults()
    freq = rebalance or strategy.rebalance
    close, open_ = panels.close, panels.open
    idx = close.index
    n, k = close.shape
    if n < warmup + 20:
        raise ValueError(f"Histórico curto: {n} pregões (mínimo {warmup + 20}).")

    has_price = close.notna()
    eligible = has_price if not dynamic_top else (liquidity_mask(panels, dynamic_top) & has_price)
    dates = rebalance_dates(idx, freq)
    dates = dates[dates >= idx[min(warmup, n - 1)]] if strategy.rebalance != "D" else dates
    targets = strategy.target_weights(close, dates, eligible, **params).reindex(columns=close.columns).fillna(0.0)
    if not strategy.long_short:
        targets = targets.clip(lower=0.0)
    gross = targets.abs().sum(axis=1)
    targets = targets.div(gross.where(gross > 1, 1.0), axis=0)  # nunca mais que 100% investido
    if strategy.rebalance == "D":  # só ordens quando o alvo muda (pares)
        changed = targets.diff().abs().sum(axis=1) > 1e-9
        changed.iloc[0] = bool(targets.iloc[0].abs().sum() > 0)
        targets = targets[changed]

    o = open_.to_numpy(dtype=float)
    c_ff = close.ffill().to_numpy(dtype=float)
    o_exec = np.where(np.isfinite(o), o, c_ff)  # sem negócio na abertura: usa o último fechamento
    pos = {d: i for i, d in enumerate(idx)}
    exec_at = {pos[d] + 1: targets.loc[d].to_numpy() for d in targets.index if pos[d] + 1 < n}
    growth = cash_returns(config.cash_index, idx)

    # a carteira começa na véspera do 1º rebalanceamento; antes disso é só aquecimento dos indicadores
    start = (min(exec_at) - 1) if exec_at else 0
    cash, shares = float(config.initial_capital), np.zeros(k)
    equity = np.empty(n)
    wts = np.zeros((n, k))
    turnover, costs = {}, 0.0
    for i in range(n):
        if i > start:
            cash *= 1.0 + growth[i]
        if i in exec_at:
            px = np.nan_to_num(o_exec[i], nan=0.0)
            eq_open = cash + float(np.dot(shares, px))
            tradable = px > 0
            target_val = np.where(tradable, exec_at[i] * max(eq_open, 0.0), shares * px)
            desired = np.divide(target_val, px, out=shares.copy(), where=tradable)
            delta = desired - shares
            notional = np.abs(delta) * px
            slip = notional * config.slippage_pct
            fee = notional * config.commission_pct + config.commission_fixed * int((np.abs(delta) > 1e-9).sum())
            cash -= float(np.dot(delta, px)) + float(slip.sum()) + float(fee if np.isscalar(fee) else fee.sum())
            costs += float(slip.sum()) + float(np.sum(fee))
            shares = desired
            turnover[idx[i]] = float(notional.sum() / max(eq_open, 1e-9) / 2)
        mark = np.nan_to_num(c_ff[i], nan=0.0)
        equity[i] = cash + float(np.dot(shares, mark))
        wts[i] = shares * mark / equity[i] if equity[i] > 0 else 0.0
        if progress and i % 250 == 0:
            progress(i, n)

    eq = pd.Series(equity, index=idx, name=strategy.name).iloc[start:]
    ppy = periods_per_year(eq.index)
    metrics = compute_metrics(eq, ppy, risk_free=config.risk_free)
    to = pd.Series(turnover)
    years = (len(eq) - 1) / ppy
    metrics["turnover_year"] = float(to.sum() / years) if years > 0 else 0.0
    metrics["rebalances"] = int(len(to))
    metrics["avg_positions"] = float((np.abs(wts[start:]) > 1e-4).sum(axis=1).mean())
    metrics["costs"] = costs
    res = PortfolioResult(equity=eq, weights=pd.DataFrame(wts, index=idx, columns=close.columns).iloc[start:],
                          targets=targets, turnover=to, costs=costs, metrics=metrics)
    return res


def add_benchmarks(res: PortfolioResult, config: BacktestConfig, ibov: pd.Series | None = None,
                   equal_weight: pd.Series | None = None) -> None:
    """Curvas de referência no mesmo período e capital: CDI, Ibovespa e igual peso."""
    eq = res.equity
    ppy = periods_per_year(eq.index)
    curves = {}
    if config.cash_index is not None:
        g = cash_returns(config.cash_index, eq.index)
        g[0] = 0.0
        curves["CDI"] = pd.Series(config.initial_capital * np.cumprod(1 + g), index=eq.index)
    if ibov is not None and len(ibov):
        s = ibov.reindex(ibov.index.union(eq.index)).ffill().reindex(eq.index)
        if s.notna().iloc[0]:
            curves["Ibovespa"] = config.initial_capital * s / s.iloc[0]
    if equal_weight is not None and len(equal_weight):
        e = equal_weight.reindex(eq.index).ffill()
        curves["Igual peso"] = config.initial_capital * e / e.iloc[0]
    res.benchmarks = curves
    res.benchmark_metrics = {k: compute_metrics(v.dropna(), ppy, risk_free=config.risk_free) for k, v in curves.items()}
