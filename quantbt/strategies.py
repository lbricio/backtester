"""Estratégias.

Uma estratégia recebe o DataFrame OHLCV e parâmetros, e devolve uma Series com a
posição alvo em cada candle: 1 = comprado, 0 = fora, -1 = vendido (frações são
permitidas). O sinal do candle t é executado na abertura de t+1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Param:
    name: str
    default: float
    min: float
    max: float
    step: float
    label: str = ""

    @property
    def is_int(self) -> bool:
        return all(float(v).is_integer() for v in (self.default, self.min, self.max, self.step))


class Strategy:
    name: str = "Estratégia"
    description: str = ""
    params: list[Param] = []

    def defaults(self) -> dict:
        return {p.name: (int(p.default) if p.is_int else p.default) for p in self.params}

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.Series:
        raise NotImplementedError


# ---------------------------------------------------------------- indicadores
def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(int(n), min_periods=int(n)).mean()


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=int(n), adjust=False, min_periods=int(n)).mean()


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = 100 - 100 / (1 + up / down)
    # sem nenhuma queda na janela o RSI é 100 (a divisão por zero daria vazio)
    return out.where(~((down == 0) & up.notna()), 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def macd(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    """Linha MACD (EMA rápida − EMA lenta) e linha de sinal (EMA da MACD)."""
    line = ema(s, fast) - ema(s, slow)
    return line, line.ewm(span=int(signal), adjust=False, min_periods=int(signal)).mean()


def stochastic(df: pd.DataFrame, n: int = 14, smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    """Estocástico: %K = posição do fechamento na faixa máxima-mínima de n candles (0–100) e %D (média de %K)."""
    lo = df["low"].rolling(int(n), min_periods=int(n)).min()
    hi = df["high"].rolling(int(n), min_periods=int(n)).max()
    k = 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)
    return k, k.rolling(int(smooth), min_periods=int(smooth)).mean()


def adx(df: pd.DataFrame, n: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """ADX (força da tendência) e +DI/−DI (direção), pelo método de Wilder."""
    up, down = df["high"].diff(), -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = 1 / int(n)
    tr = atr(df, n)
    plus_di = 100 * plus_dm.ewm(alpha=a, adjust=False, min_periods=int(n)).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False, min_periods=int(n)).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False, min_periods=int(n)).mean(), plus_di, minus_di


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0) -> pd.Series:
    """Direção do Supertrend: 1 (alta) ou -1 (baixa); NaN no aquecimento."""
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    a = atr(df, n).to_numpy()
    c = df["close"].to_numpy()
    up_b, lo_b = hl2 + mult * a, hl2 - mult * a
    out = np.full(len(df), np.nan)
    fu, fl, trend = np.nan, np.nan, 1
    for i in range(len(df)):
        if np.isnan(a[i]):
            continue
        if np.isnan(fu):  # primeiro candle válido
            fu, fl = up_b[i], lo_b[i]
        else:
            fu = up_b[i] if (up_b[i] < fu or c[i - 1] > fu) else fu
            fl = lo_b[i] if (lo_b[i] > fl or c[i - 1] < fl) else fl
            if trend == 1 and c[i] < fl:
                trend = -1
            elif trend == -1 and c[i] > fu:
                trend = 1
        out[i] = trend
    return pd.Series(out, index=df.index)


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_af: float = 0.2) -> pd.Series:
    """Direção do Parabolic SAR de Wilder: 1 (alta) ou -1 (baixa)."""
    h, lo = df["high"].to_numpy(), df["low"].to_numpy()
    n = len(df)
    out = np.full(n, np.nan)
    if n < 3:
        return pd.Series(out, index=df.index)
    trend, af = 1, step
    ep, sar = h[0], lo[0]
    for i in range(1, n):
        sar = sar + af * (ep - sar)
        if trend == 1:
            sar = min(sar, lo[i - 1], lo[i - 2] if i > 1 else lo[i - 1])
            if lo[i] < sar:  # virou para baixa
                trend, sar, ep, af = -1, ep, lo[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, max_af)
        else:
            sar = max(sar, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > sar:  # virou para alta
                trend, sar, ep, af = 1, ep, h[i], step
            elif lo[i] < ep:
                ep, af = lo[i], min(af + step, max_af)
        out[i] = trend
    return pd.Series(out, index=df.index)


def _hold(entries: pd.Series, exits: pd.Series, value: float = 1.0) -> pd.Series:
    """Converte eventos de entrada/saída em posição mantida."""
    state = pd.Series(np.nan, index=entries.index)
    state[exits.fillna(False)] = 0.0
    state[entries.fillna(False)] = value
    return state.ffill().fillna(0.0)


# ---------------------------------------------------------------- built-ins
class BuyAndHold(Strategy):
    name = "Buy & Hold"
    description = "Compra no primeiro candle e mantém até o fim."
    params = []

    def generate_signals(self, df, **_):
        return pd.Series(1.0, index=df.index)


class SmaCross(Strategy):
    name = "Cruzamento de médias"
    description = "Comprado quando a média rápida está acima da lenta; vendido (ou fora) caso contrário."
    params = [
        Param("fast", 20, 5, 50, 5, "Média rápida"),
        Param("slow", 50, 20, 200, 20, "Média lenta"),
        Param("short", 0, 0, 1, 1, "Operar vendido (0/1)"),
    ]

    def generate_signals(self, df, fast=20, slow=50, short=0):
        f, s = sma(df["close"], fast), sma(df["close"], slow)
        sig = pd.Series(np.where(f > s, 1.0, -1.0 if short else 0.0), index=df.index)
        return sig.where(s.notna(), 0.0)


class RsiReversion(Strategy):
    name = "Reversão por RSI"
    description = "Compra quando o RSI fica abaixo do nível de sobrevenda e sai quando passa do nível de saída."
    params = [
        Param("period", 14, 2, 30, 2, "Período RSI"),
        Param("oversold", 30, 5, 50, 5, "Sobrevenda (entrada)"),
        Param("exit", 55, 40, 90, 5, "Nível de saída"),
    ]

    def generate_signals(self, df, period=14, oversold=30, exit=55):
        r = rsi(df["close"], int(period))
        return _hold(r < oversold, r > exit)


class BollingerReversion(Strategy):
    name = "Bandas de Bollinger"
    description = "Compra ao fechar abaixo da banda inferior e sai ao voltar à média."
    params = [
        Param("period", 20, 5, 100, 5, "Período"),
        Param("k", 2.0, 0.5, 4.0, 0.25, "Desvios (k)"),
    ]

    def generate_signals(self, df, period=20, k=2.0):
        c = df["close"]
        mid = sma(c, period)
        sd = c.rolling(int(period)).std()
        return _hold(c < mid - k * sd, c > mid)


class DonchianBreakout(Strategy):
    name = "Rompimento de Donchian"
    description = "Compra no rompimento da máxima de N candles; sai na perda da mínima de M candles."
    params = [
        Param("entry", 55, 10, 200, 15, "Janela de entrada"),
        Param("exit", 20, 5, 60, 5, "Janela de saída"),
    ]

    def generate_signals(self, df, entry=55, exit=20):
        upper = df["high"].rolling(int(entry)).max().shift(1)
        lower = df["low"].rolling(int(exit)).min().shift(1)
        return _hold(df["close"] > upper, df["close"] < lower)


class Momentum(Strategy):
    name = "Momentum"
    description = "Comprado quando o retorno dos últimos N candles supera o limiar."
    params = [
        Param("lookback", 120, 20, 300, 20, "Janela"),
        Param("threshold", 0.0, -0.1, 0.1, 0.02, "Limiar"),
    ]

    def generate_signals(self, df, lookback=120, threshold=0.0):
        m = df["close"].pct_change(int(lookback))
        return (m > threshold).astype(float)


def _direction(trend: pd.Series, short) -> pd.Series:
    """Converte direção (1/-1, NaN no aquecimento) em posição: comprado, vendido (se permitido) ou fora."""
    pos = trend.where(trend > 0, -1.0 if short else 0.0)
    return pos.where(trend.notna(), 0.0)


class EmaCross(Strategy):
    name = "Médias exponenciais"
    description = "Como o cruzamento de médias, mas com médias exponenciais (reagem mais rápido)."
    params = [
        Param("fast", 9, 3, 50, 3, "Média rápida (EMA)"),
        Param("slow", 21, 10, 200, 10, "Média lenta (EMA)"),
        Param("short", 0, 0, 1, 1, "Operar vendido (0/1)"),
    ]

    def generate_signals(self, df, fast=9, slow=21, short=0):
        f, s = ema(df["close"], fast), ema(df["close"], slow)
        return _direction(pd.Series(np.where(f > s, 1.0, -1.0), index=df.index).where(s.notna()), short)


class MacdCross(Strategy):
    name = "MACD"
    description = "Comprado quando a linha MACD está acima da linha de sinal (momento de alta)."
    params = [
        Param("fast", 12, 6, 20, 2, "EMA rápida"),
        Param("slow", 26, 20, 50, 4, "EMA lenta"),
        Param("signal", 9, 5, 15, 2, "Linha de sinal"),
        Param("short", 0, 0, 1, 1, "Operar vendido (0/1)"),
    ]

    def generate_signals(self, df, fast=12, slow=26, signal=9, short=0):
        line, sig = macd(df["close"], fast, slow, signal)
        return _direction(pd.Series(np.where(line > sig, 1.0, -1.0), index=df.index).where(sig.notna()), short)


class TrendFilter(Strategy):
    name = "Acima da média de 200"
    description = "Comprado só quando o preço fecha acima da média de N dias (o clássico filtro de tendência)."
    params = [Param("period", 200, 50, 300, 25, "Média (dias)")]

    def generate_signals(self, df, period=200):
        m = sma(df["close"], period)
        return (df["close"] > m).astype(float).where(m.notna(), 0.0)


class AdxTrend(Strategy):
    name = "ADX / DMI"
    description = "Comprado quando há tendência forte (ADX acima do limite) e de alta (+DI acima de −DI)."
    params = [
        Param("period", 14, 7, 30, 7, "Período"),
        Param("threshold", 25, 15, 40, 5, "ADX mínimo"),
        Param("short", 0, 0, 1, 1, "Operar vendido (0/1)"),
    ]

    def generate_signals(self, df, period=14, threshold=25, short=0):
        a, pdi, mdi = adx(df, period)
        strong = a > threshold
        pos = pd.Series(0.0, index=df.index)
        pos[strong & (pdi > mdi)] = 1.0
        if short:
            pos[strong & (mdi > pdi)] = -1.0
        return pos


class Supertrend(Strategy):
    name = "Supertrend"
    description = "Segue a tendência com uma faixa baseada no ATR; troca de lado quando o preço rompe a faixa."
    params = [
        Param("period", 10, 5, 30, 5, "ATR (dias)"),
        Param("mult", 3.0, 1.0, 5.0, 0.5, "Multiplicador do ATR"),
        Param("short", 0, 0, 1, 1, "Operar vendido (0/1)"),
    ]

    def generate_signals(self, df, period=10, mult=3.0, short=0):
        return _direction(supertrend(df, int(period), float(mult)), short)


class ParabolicSar(Strategy):
    name = "Parabolic SAR"
    description = "Stop móvel que acelera com a tendência (Wilder); comprado enquanto o preço está acima do SAR."
    params = [
        Param("step", 0.02, 0.01, 0.05, 0.01, "Aceleração"),
        Param("max_af", 0.2, 0.1, 0.4, 0.1, "Aceleração máxima"),
        Param("short", 0, 0, 1, 1, "Operar vendido (0/1)"),
    ]

    def generate_signals(self, df, step=0.02, max_af=0.2, short=0):
        return _direction(parabolic_sar(df, float(step), float(max_af)), short)


class BollingerBreakout(Strategy):
    name = "Rompimento de Bollinger"
    description = "Compra quando o preço fecha acima da banda superior (força) e sai ao voltar para a média."
    params = [
        Param("period", 20, 10, 60, 5, "Período"),
        Param("k", 2.0, 1.0, 3.0, 0.5, "Desvios (k)"),
    ]

    def generate_signals(self, df, period=20, k=2.0):
        c = df["close"]
        mid = sma(c, period)
        sd = c.rolling(int(period)).std()
        return _hold(c > mid + k * sd, c < mid)


class Setup91(Strategy):
    name = "Setup 9.1 (Larry Williams)"
    description = "Compra quando a média exponencial de 9 vira para cima e sai quando ela vira para baixo."
    params = [Param("period", 9, 5, 21, 2, "Média exponencial")]

    def generate_signals(self, df, period=9):
        e = ema(df["close"], period)
        slope = e.diff()
        turned_up = (slope > 0) & (slope.shift(1) <= 0)
        turned_down = (slope < 0) & (slope.shift(1) >= 0)
        return _hold(turned_up, turned_down)


class ConnorsRsi2(Strategy):
    name = "RSI(2) de Connors"
    description = ("Reversão de curto prazo: compra quedas fortes (RSI de 2 dias muito baixo) só em tendência "
                   "de alta (acima da média de 200) e sai quando o preço passa a média curta.")
    params = [
        Param("entry", 10, 2, 30, 2, "RSI(2) de entrada"),
        Param("trend", 200, 100, 300, 50, "Média de tendência"),
        Param("exit_ma", 5, 3, 20, 1, "Média de saída"),
    ]

    def generate_signals(self, df, entry=10, trend=200, exit_ma=5):
        c = df["close"]
        r = rsi(c, 2)
        long_trend = c > sma(c, trend)
        return _hold((r < entry) & long_trend, c > sma(c, exit_ma))


class StochasticReversion(Strategy):
    name = "Estocástico"
    description = "Compra quando o estocástico sai da sobrevenda (cruza para cima do nível baixo) e sai na sobrecompra."
    params = [
        Param("period", 14, 5, 30, 3, "Período"),
        Param("oversold", 20, 5, 40, 5, "Sobrevenda"),
        Param("overbought", 80, 60, 95, 5, "Sobrecompra (saída)"),
    ]

    def generate_signals(self, df, period=14, oversold=20, overbought=80):
        k, _ = stochastic(df, period)
        enter = (k > oversold) & (k.shift(1) <= oversold)
        return _hold(enter, k > overbought)


class TurnOfMonth(Strategy):
    name = "Virada do mês"
    description = ("Efeito sazonal: comprado nos últimos pregões de cada mês e nos primeiros do mês seguinte, "
                   "quando entram fluxos de salários e fundos. Fora no resto do tempo.")
    params = [
        Param("before", 1, 0, 5, 1, "Pregões antes da virada"),
        Param("after", 3, 1, 8, 1, "Pregões depois da virada"),
    ]

    def generate_signals(self, df, before=1, after=3):
        idx = pd.DatetimeIndex(df.index)
        month = pd.Series(idx.to_period("M"), index=idx)
        pos_from_start = month.groupby(month).cumcount()  # 0 = 1º pregão do mês
        pos_from_end = month.groupby(month).cumcount(ascending=False)  # 0 = último pregão
        hold = (pos_from_start < int(after)) | (pos_from_end < int(before))
        # o sinal do candle t vale para o candle t+1: usa o calendário (conhecido de antemão), não preços
        return hold.astype(float).shift(-1).fillna(0.0)


# ---------------------------------------------------------------- institucionais (um ativo)
def period_end_mask(index: pd.Index, freq: str = "W-FRI") -> pd.Series:
    """True no último pregão de cada período (semana, mês...). Usa só o calendário."""
    idx = pd.DatetimeIndex(index)
    per = pd.Series(idx.to_period(freq[0] if freq.startswith(("M", "Q")) else freq), index=idx)
    return per != per.shift(-1)


def rebalance_weights(w: pd.Series, freq: str = "W-FRI", step: float = 0.1) -> pd.Series:
    """Arredonda o peso em degraus e só o altera no fim de cada período (evita giro e custos diários)."""
    q = (w / step).round() * step
    return q.where(period_end_mask(w.index, freq)).ffill().fillna(0.0)


def realized_vol(close: pd.Series, n: int = 63, ppy: float = 252.0) -> pd.Series:
    r = np.log(close / close.shift(1))
    return r.rolling(int(n), min_periods=int(n)).std() * np.sqrt(ppy)


class TrendVolTarget(Strategy):
    name = "Tendência com alvo de volatilidade (CTA)"
    description = ("Momentum de série temporal 12-1 (Moskowitz, Ooi e Pedersen), como os fundos de "
                   "seguimento de tendência: comprado se o retorno de 12 meses (sem o último) é positivo, com "
                   "tamanho de posição ajustado para uma volatilidade-alvo. Rebalanceia semanalmente.")
    params = [
        Param("lookback", 252, 63, 504, 63, "Janela do momentum"),
        Param("skip", 21, 0, 63, 21, "Pular os últimos (dias)"),
        Param("target_vol", 0.15, 0.05, 0.40, 0.05, "Volatilidade-alvo (a.a.)"),
        Param("vol_window", 63, 21, 252, 21, "Janela da volatilidade"),
        Param("short", 0, 0, 1, 1, "Operar vendido (0/1)"),
    ]

    def generate_signals(self, df, lookback=252, skip=21, target_vol=0.15, vol_window=63, short=0):
        c = df["close"]
        mom = c.shift(int(skip)) / c.shift(int(lookback)) - 1
        direction = np.sign(mom)
        if not short:
            direction = direction.clip(lower=0)
        size = (target_vol / realized_vol(c, vol_window)).clip(upper=1.0)
        return rebalance_weights((direction * size).fillna(0.0))


class VolTargetHold(Strategy):
    name = "Buy & Hold com alvo de volatilidade"
    description = ("Sempre comprado, mas reduz a posição quando a volatilidade sobe (o resto fica no caixa). "
                   "Técnica comum em fundos para estabilizar o risco. Rebalanceia semanalmente.")
    params = [
        Param("target_vol", 0.20, 0.05, 0.40, 0.05, "Volatilidade-alvo (a.a.)"),
        Param("vol_window", 63, 21, 252, 21, "Janela da volatilidade"),
    ]

    def generate_signals(self, df, target_vol=0.20, vol_window=63):
        size = (target_vol / realized_vol(df["close"], vol_window)).clip(upper=1.0)
        return rebalance_weights(size.fillna(0.0))


class MonthlyTiming(Strategy):
    name = "Timing de 10 meses (Faber)"
    description = ("No último pregão de cada mês: comprado se o preço está acima da média dos últimos 10 "
                   "fechamentos mensais, senão no caixa. Base de muitas carteiras de alocação tática (GTAA).")
    params = [Param("months", 10, 3, 24, 1, "Média (meses)")]

    def generate_signals(self, df, months=10):
        c = df["close"]
        is_end = period_end_mask(c.index, "M")
        monthly = c[is_end]
        decision = (monthly > monthly.rolling(int(months), min_periods=int(months)).mean()).astype(float)
        decision[monthly.rolling(int(months), min_periods=int(months)).mean().isna()] = 0.0
        return decision.reindex(c.index).ffill().fillna(0.0)


class TurtleAtr(Strategy):
    name = "Tartarugas (Turtle) com risco por ATR"
    description = ("Sistema das Tartarugas de Richard Dennis: entra no rompimento da máxima de N dias, sai na "
                   "mínima de M dias, e dimensiona a posição para arriscar uma fração fixa do capital "
                   "(stop de k × ATR). Posição máxima de 100%.")
    params = [
        Param("entry", 20, 10, 60, 5, "Janela de entrada"),
        Param("exit", 10, 5, 30, 5, "Janela de saída"),
        Param("risk", 0.01, 0.005, 0.03, 0.005, "Risco por operação"),
        Param("stop_atr", 2.0, 1.0, 4.0, 0.5, "Stop (× ATR)"),
    ]

    def generate_signals(self, df, entry=20, exit=10, risk=0.01, stop_atr=2.0):
        c = df["close"].to_numpy()
        upper = df["high"].rolling(int(entry)).max().shift(1).to_numpy()
        lower = df["low"].rolling(int(exit)).min().shift(1).to_numpy()
        a = atr(df, 20).to_numpy()
        out = np.zeros(len(df))
        size = 0.0
        for i in range(len(df)):
            if size == 0.0:
                if np.isfinite(upper[i]) and np.isfinite(a[i]) and a[i] > 0 and c[i] > upper[i]:
                    size = min(risk / (stop_atr * a[i] / c[i]), 1.0)  # tamanho fixado na entrada
            elif np.isfinite(lower[i]) and c[i] < lower[i]:
                size = 0.0
            out[i] = size
        return pd.Series(np.round(out, 2), index=df.index)


# ---------------------------------------------------------------- clássicas simples (mais opções)
class TripleMA(Strategy):
    name = "Cruzamento triplo de médias"
    description = "Comprado quando as três médias estão alinhadas para cima (curta > média > longa)."
    params = [Param("fast", 5, 3, 20, 1, "Média curta"), Param("mid", 20, 10, 60, 5, "Média intermediária"),
              Param("slow", 60, 30, 200, 10, "Média longa")]

    def generate_signals(self, df, fast=5, mid=20, slow=60):
        c = df["close"]
        f, m, s = sma(c, fast), sma(c, mid), sma(c, slow)
        return ((f > m) & (m > s)).astype(float).where(s.notna(), 0.0)


class PriceCrossMA(Strategy):
    name = "Preço cruza a média"
    description = "Compra quando o preço fecha acima da média de N dias e sai quando fecha abaixo."
    params = [Param("period", 50, 10, 200, 10, "Média (dias)"), Param("short", 0, 0, 1, 1, "Operar vendido (0/1)")]

    def generate_signals(self, df, period=50, short=0):
        c, m = df["close"], sma(df["close"], period)
        return _direction(pd.Series(np.where(c > m, 1.0, -1.0), index=df.index).where(m.notna()), short)


class Ichimoku(Strategy):
    name = "Ichimoku (nuvem)"
    description = "Comprado quando o preço está acima da nuvem e a Tenkan está acima da Kijun."
    params = [Param("tenkan", 9, 5, 20, 1, "Tenkan"), Param("kijun", 26, 15, 60, 1, "Kijun"),
              Param("senkou", 52, 30, 120, 2, "Senkou B")]

    def generate_signals(self, df, tenkan=9, kijun=26, senkou=52):
        h, lo, c = df["high"], df["low"], df["close"]
        mid = lambda n: (h.rolling(int(n)).max() + lo.rolling(int(n)).min()) / 2  # noqa: E731
        t, k = mid(tenkan), mid(kijun)
        # a nuvem de hoje foi calculada `kijun` candles atrás (deslocada para frente): só usa o passado
        span_a = ((t + k) / 2).shift(int(kijun))
        span_b = mid(senkou).shift(int(kijun))
        top = pd.concat([span_a, span_b], axis=1).max(axis=1, skipna=False)
        return ((c > top) & (t > k)).astype(float).where(top.notna(), 0.0)


class KeltnerBreakout(Strategy):
    name = "Rompimento de Keltner"
    description = "Compra quando o preço fecha acima do canal de Keltner (média + k × ATR) e sai na média."
    params = [Param("period", 20, 10, 50, 5, "Média (EMA)"), Param("mult", 2.0, 1.0, 3.0, 0.5, "Multiplicador do ATR")]

    def generate_signals(self, df, period=20, mult=2.0):
        c = df["close"]
        mid = ema(c, period)
        return _hold(c > mid + mult * atr(df, period), c < mid)


class Envelope(Strategy):
    name = "Envelope de médias"
    description = "Compra quando o preço cai X% abaixo da média e sai quando volta para a média."
    params = [Param("period", 20, 10, 100, 5, "Média (dias)"), Param("pct", 0.05, 0.02, 0.15, 0.01, "Distância (%)")]

    def generate_signals(self, df, period=20, pct=0.05):
        c, m = df["close"], sma(df["close"], period)
        return _hold(c < m * (1 - pct), c > m)


class WilliamsR(Strategy):
    name = "Williams %R"
    description = "Compra quando o %R sai da sobrevenda (cruza para cima de −80) e sai acima de −20."
    params = [Param("period", 14, 5, 30, 1, "Período"), Param("oversold", -80, -95, -60, 5, "Sobrevenda"),
              Param("overbought", -20, -40, -5, 5, "Sobrecompra (saída)")]

    def generate_signals(self, df, period=14, oversold=-80, overbought=-20):
        k, _ = stochastic(df, period)
        wr = k - 100  # %R = %K − 100 (vai de −100 a 0)
        return _hold((wr > oversold) & (wr.shift(1) <= oversold), wr > overbought)


class CciReversion(Strategy):
    name = "CCI"
    description = "Commodity Channel Index: compra quando o CCI volta para cima de −100 e sai acima de +100."
    params = [Param("period", 20, 10, 50, 5, "Período"), Param("level", 100, 50, 200, 25, "Nível")]

    def generate_signals(self, df, period=20, level=100):
        tp = (df["high"] + df["low"] + df["close"]) / 3
        m = tp.rolling(int(period)).mean()
        md = tp.rolling(int(period)).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        cci = (tp - m) / (0.015 * md.replace(0, np.nan))
        return _hold((cci > -level) & (cci.shift(1) <= -level), cci > level)


class AroonTrend(Strategy):
    name = "Aroon"
    description = "Comprado quando o Aroon Up está acima do Aroon Down e acima de 70 (máximas recentes)."
    params = [Param("period", 25, 10, 60, 5, "Período"), Param("level", 70, 50, 90, 10, "Nível mínimo")]

    def generate_signals(self, df, period=25, level=70):
        n = int(period)
        up = df["high"].rolling(n + 1).apply(lambda x: 100 * x.argmax() / n, raw=True)
        down = df["low"].rolling(n + 1).apply(lambda x: 100 * x.argmin() / n, raw=True)
        return ((up > down) & (up > level)).astype(float).where(up.notna(), 0.0)


class High52w(Strategy):
    name = "Máxima de 52 semanas"
    description = "Compra quando o preço fecha na máxima de 1 ano e sai quando cai X% abaixo dessa máxima."
    params = [Param("window", 252, 126, 504, 63, "Janela (dias)"), Param("trail", 0.10, 0.05, 0.30, 0.05, "Saída (% abaixo)")]

    def generate_signals(self, df, window=252, trail=0.10):
        c = df["close"]
        hi = c.rolling(int(window), min_periods=int(window)).max()
        return _hold(c >= hi, c < hi * (1 - trail))


class ConsecutiveDown(Strategy):
    name = "Quedas consecutivas"
    description = "Compra depois de N fechamentos seguidos em queda e sai no primeiro fechamento em alta."
    params = [Param("days", 3, 2, 7, 1, "Quedas seguidas")]

    def generate_signals(self, df, days=3):
        down = df["close"].diff() < 0
        streak = down.astype(int).groupby((~down).cumsum()).cumsum()
        return _hold(streak >= int(days), df["close"].diff() > 0)


class HeikinAshi(Strategy):
    name = "Heikin-Ashi"
    description = "Comprado depois de N candles Heikin-Ashi de alta seguidos; sai no primeiro candle de baixa."
    params = [Param("confirm", 2, 1, 5, 1, "Candles de confirmação")]

    def generate_signals(self, df, confirm=2):
        o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        ha_c = (o + h + lo + c) / 4
        ha_o = np.empty(len(df))
        ha_o[0] = (o[0] + c[0]) / 2
        for i in range(1, len(df)):
            ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2
        bull = pd.Series(ha_c > ha_o, index=df.index)
        run = bull.astype(int).groupby((~bull).cumsum()).cumsum()
        return _hold(run >= int(confirm), ~bull)


class ObvTrend(Strategy):
    name = "OBV (volume)"
    description = "On-Balance Volume: comprado quando o OBV está acima da sua média (volume confirmando a alta)."
    params = [Param("period", 20, 10, 100, 5, "Média do OBV")]

    def generate_signals(self, df, period=20):
        obv = (np.sign(df["close"].diff()).fillna(0) * df["volume"]).cumsum()
        m = obv.rolling(int(period)).mean()
        if df["volume"].abs().sum() == 0:
            return pd.Series(0.0, index=df.index)  # ativo sem volume (ex.: alguns externos)
        return (obv > m).astype(float).where(m.notna(), 0.0)


class ChandelierExit(Strategy):
    name = "Saída Chandelier (stop por ATR)"
    description = ("Entra quando o preço passa a média de N dias e sai pelo stop Chandelier: máxima recente "
                   "menos k × ATR (stop móvel que só sobe).")
    params = [Param("period", 22, 10, 60, 2, "Janela"), Param("mult", 3.0, 1.5, 5.0, 0.5, "Multiplicador do ATR")]

    def generate_signals(self, df, period=22, mult=3.0):
        c = df["close"]
        stop = df["high"].rolling(int(period)).max() - mult * atr(df, period)
        return _hold((c > sma(c, period)) & (c > stop), c < stop)


class RsiMomentum(Strategy):
    name = "IFR cruzando 50"
    description = "Usa o RSI como momento: comprado quando o RSI está acima de 50 (força compradora)."
    params = [Param("period", 14, 5, 30, 1, "Período"), Param("level", 50, 40, 60, 5, "Nível"),
              Param("short", 0, 0, 1, 1, "Operar vendido (0/1)")]

    def generate_signals(self, df, period=14, level=50, short=0):
        r = rsi(df["close"], int(period))
        return _direction(pd.Series(np.where(r > level, 1.0, -1.0), index=df.index).where(r.notna()), short)


class Setup123(Strategy):
    name = "Setup 123 de compra"
    description = ("Padrão de três candles: o do meio tem a menor mínima (fundo). Compra quando o preço "
                   "rompe a máxima do 3º candle; sai quando perde a mínima do fundo ou após N candles.")
    params = [Param("max_hold", 10, 3, 30, 1, "Saída por tempo (candles)")]

    def generate_signals(self, df, max_hold=10):
        h, lo, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
        out = np.zeros(len(df))
        trigger = stop = np.nan
        held = 0
        for i in range(3, len(df)):
            if held:
                held += 1
                if c[i] < stop or held > max_hold:
                    held = 0
                out[i] = 1.0 if held else 0.0
                continue
            # padrão formado nos candles i-3, i-2, i-1 (fundo no do meio): gatilho na máxima do 3º
            if lo[i - 2] < lo[i - 3] and lo[i - 2] < lo[i - 1]:
                trigger, stop = h[i - 1], lo[i - 2]
            if np.isfinite(trigger) and c[i] > trigger:
                held, trigger = 1, np.nan
                out[i] = 1.0
        return pd.Series(out, index=df.index)


BUILTIN: list[Strategy] = [
    # tendência
    SmaCross(), EmaCross(), TripleMA(), PriceCrossMA(), TrendFilter(), MacdCross(), DonchianBreakout(),
    BollingerBreakout(), KeltnerBreakout(), High52w(), AdxTrend(), AroonTrend(), Ichimoku(), Supertrend(),
    ParabolicSar(), ChandelierExit(), HeikinAshi(), Setup91(), Setup123(), RsiMomentum(), ObvTrend(), Momentum(),
    # reversão à média
    RsiReversion(), ConnorsRsi2(), BollingerReversion(), Envelope(), StochasticReversion(), WilliamsR(),
    CciReversion(), ConsecutiveDown(),
    # institucionais (gestão de risco e alocação tática)
    TrendVolTarget(), VolTargetHold(), MonthlyTiming(), TurtleAtr(),
    # sazonalidade e referência
    TurnOfMonth(), BuyAndHold(),
]


# ---------------------------------------------------------------- personalizada
CUSTOM_TEMPLATE = '''\
# Estratégia personalizada
#
# df: DataFrame com colunas open, high, low, close, volume (índice = datas)
# Retorne uma Series com a posição desejada em cada candle:
#   1 = comprado, 0 = fora, -1 = vendido (frações como 0.5 são permitidas)
# O sinal calculado no fechamento do candle t é executado na abertura de t+1.
#
# Disponíveis: pd, np, sma, ema, rsi, atr, macd, stochastic, adx, supertrend, parabolic_sar
#
# PARAMS define os parâmetros editáveis/otimizáveis:
#   "nome": valor  ou  "nome": {"default":..., "min":..., "max":..., "step":...}

NAME = "Tendência com filtro de volatilidade"

PARAMS = {
    "trend": {"default": 100, "min": 20, "max": 300, "step": 20},
    "atr_period": {"default": 14, "min": 5, "max": 50, "step": 5},
    "max_atr_pct": {"default": 0.03, "min": 0.01, "max": 0.06, "step": 0.01},
}


def generate_signals(df, trend, atr_period, max_atr_pct):
    close = df["close"]
    above_trend = close > sma(close, trend)
    calm = atr(df, atr_period) / close < max_atr_pct
    return (above_trend & calm).astype(float)
'''


class CodeStrategy(Strategy):
    """Estratégia definida por código Python do usuário."""

    def __init__(self, source: str, name: str | None = None) -> None:
        self.source = source
        ns: dict = {
            "pd": pd, "np": np, "sma": sma, "ema": ema, "rsi": rsi, "atr": atr, "macd": macd,
            "stochastic": stochastic, "adx": adx, "supertrend": supertrend, "parabolic_sar": parabolic_sar,
            "__name__": "quantbt_custom",
        }
        exec(compile(source, "<estratégia>", "exec"), ns)  # código do próprio usuário
        fn: Callable | None = ns.get("generate_signals")
        if not callable(fn):
            raise ValueError("O código precisa definir a função generate_signals(df, ...).")
        self._fn = fn
        self.name = name or str(ns.get("NAME", "Personalizada"))
        self.description = "Estratégia personalizada (editor de código)."
        self.params = [self._parse_param(k, v) for k, v in dict(ns.get("PARAMS", {})).items()]

    @staticmethod
    def _parse_param(name: str, spec) -> Param:
        if isinstance(spec, dict):
            d = float(spec.get("default", 0))
            return Param(
                name, d,
                float(spec.get("min", min(0.0, d))),
                float(spec.get("max", max(d * 5, 1.0))),
                float(spec.get("step", 1 if float(d).is_integer() else abs(d) / 10 or 0.1)),
                str(spec.get("label", name)),
            )
        d = float(spec)
        if d.is_integer():
            return Param(name, d, 1, max(d * 5, 10), 1, name)
        return Param(name, d, min(0.0, d * 5), max(d * 5, 1.0), abs(d) / 10 or 0.1, name)

    def generate_signals(self, df, **params):
        out = self._fn(df.copy(), **params)
        if not isinstance(out, pd.Series):
            out = pd.Series(np.asarray(out, dtype=float), index=df.index)
        return out.astype(float)


def user_strategies_dir() -> Path:
    d = Path.home() / ".quantbt" / "strategies"
    d.mkdir(parents=True, exist_ok=True)
    return d
