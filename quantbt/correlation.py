"""Análise de correlação entre ativos.

Todas as medidas usam RETORNOS logarítmicos, r_t = ln(P_t / P_{t-1}), e não preços:
preços com tendência parecem correlacionados mesmo sem relação real. Cada par usa
apenas os períodos em que os dois ativos têm dados.

Métodos:
- pearson:  correlação linear clássica.
- spearman: Pearson sobre os postos (rankings) dos retornos; robusto a outliers.
- kendall:  tau de Kendall (concordância de sinais entre pares de dias); robusto, lento.
- ewma:     Pearson com pesos exponenciais (meia-vida em períodos); enfatiza o recente.
- residual: Pearson dos resíduos após remover o fator de mercado (r_i = a + b·m + e).
- stress:   Pearson apenas nos dias em que o mercado esteve entre os piores (quantil).
- sign:     concordância de direção: (dias na mesma direção − dias opostos) / dias com
            movimento dos dois; ignora a intensidade. % de dias na mesma direção = (1 + valor) / 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

FREQS = {"Diária": None, "Semanal": "W-FRI", "Mensal": "ME"}

METHODS: dict[str, str] = {
    "Direção (sinais)": "sign",
    "Pearson": "pearson",
    "Spearman (postos)": "spearman",
    "Kendall (tau)": "kendall",
    "Exponencial (EWMA)": "ewma",
    "Residual (sem mercado)": "residual",
    "Estresse (piores dias)": "stress",
}
METHOD_LABEL = {v: k for k, v in METHODS.items()}
METHOD_INFO: dict[str, str] = {
    "sign": "Só a direção do dia (subiu/caiu), sem a intensidade: (dias na mesma direção − dias "
            "opostos) ÷ dias em que ambos se mexeram. % de dias juntos = (1 + valor) ÷ 2; ex.: +0,6 = 80%. "
            "Dias com movimento menor que a zona neutra são ignorados.",
    "pearson": "Correlação linear dos retornos. Padrão do mercado; sensível a dias extremos.",
    "spearman": "Correlação dos postos (rankings) dos retornos. Menos sensível a outliers e a "
                "relações não lineares monotônicas.",
    "kendall": "Proporção de dias concordantes menos discordantes. A mais robusta, porém lenta: "
               "limitada a poucos ativos.",
    "ewma": "Pearson com pesos que decaem exponencialmente (meia-vida em períodos). Mostra a "
            "relação atual, dando mais peso ao passado recente.",
    "residual": "Remove o efeito do mercado (Ibovespa ou média do universo) de cada ativo e "
                "correlaciona o que sobra. Revela relações próprias: mesmo setor, holding × controlada.",
    "stress": "Pearson só nos piores dias do mercado (quantil). Mostra se a diversificação some "
              "nas quedas. Obs.: amostras condicionais tendem a subestimar a correlação.",
}
KENDALL_MAX_ASSETS = 80
DEFAULT_ROLLING_TOP = 3000


@dataclass
class MethodParams:
    halflife: float = 126.0  # EWMA: meia-vida em períodos (126 pregões ≈ 6 meses)
    tail_q: float = 0.2  # estresse: fração dos piores dias do mercado
    market: pd.Series | None = None  # preços do fator de mercado (ex.: BOVA11); None = média do universo
    market_name: str = ""
    neutral: float = 0.0  # direção: |retorno| abaixo disso conta como "parado" (ex.: 0.001 = 0,1%)


@dataclass
class CorrelationResult:
    prices: pd.DataFrame
    returns: pd.DataFrame
    matrix: pd.DataFrame
    pairs: pd.DataFrame
    dropped: list[str]
    method: str = "pearson"
    compared: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    params: MethodParams | None = None
    window: int = 60  # janela da medida móvel usada no cálculo


def to_returns(prices: pd.DataFrame | pd.Series, freq: str | None = None, log: bool = True):
    p = prices.sort_index()
    if freq:
        p = p.resample(freq).last()
    r = np.log(p / p.shift(1)) if log else p.pct_change(fill_method=None)
    return r.replace([np.inf, -np.inf], np.nan).iloc[1:]


# ---------------------------------------------------------------- núcleo vetorizado
def pairwise_corr(
    returns: pd.DataFrame,
    min_obs: int = 30,
    weights: np.ndarray | None = None,
    rows: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pearson par a par com dados faltantes, vetorizado (opcionalmente ponderado).

    weights: peso de cada período (EWMA). rows: máscara booleana de períodos a usar.
    Retorna (correlação, nº de observações em comum).
    """
    x = returns.to_numpy(dtype=np.float64)
    if rows is not None:
        x = x[rows]
        weights = None if weights is None else weights[rows]
    m = np.isfinite(x).astype(np.float64)
    x = np.where(m > 0, x, 0.0)
    count = m.T @ m
    mw = m if weights is None else m * weights[:, None]
    xw = x if weights is None else x * weights[:, None]
    n = mw.T @ m  # soma dos pesos nos períodos em comum
    sx = xw.T @ m  # soma de x(i) nos períodos em comum com j
    sy = sx.T
    sxx = (xw * x).T @ m
    syy = sxx.T
    sxy = xw.T @ x
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = n * sxy - sx * sy
        var = (n * sxx - sx * sx) * (n * syy - sy * sy)
        c = cov / np.sqrt(var)
    c[(count < min_obs) | ~np.isfinite(c)] = np.nan
    np.clip(c, -1.0, 1.0, out=c)
    cols = returns.columns
    return pd.DataFrame(c, index=cols, columns=cols), pd.DataFrame(count.astype(np.int64), index=cols, columns=cols)


def sign_concordance(returns: pd.DataFrame, min_obs: int = 30, neutral: float = 0.0):
    """Concordância de direção par a par. Retorna (valor em [-1, 1], nº de dias com movimento dos dois)."""
    x = returns.to_numpy(dtype=np.float64)
    s = np.where(np.isfinite(x) & (np.abs(x) > neutral), np.sign(x), 0.0)
    nz = (s != 0).astype(np.float64)
    count = nz.T @ nz
    with np.errstate(invalid="ignore", divide="ignore"):
        c = (s.T @ s) / count
    c[(count < min_obs) | ~np.isfinite(c)] = np.nan
    cols = returns.columns
    return pd.DataFrame(c, index=cols, columns=cols), pd.DataFrame(count.astype(np.int64), index=cols, columns=cols)


def market_agreement(rets: pd.DataFrame, m: pd.Series, neutral: float = 0.0) -> pd.Series:
    """Fração dos dias (com movimento) em que cada ativo foi na mesma direção do mercado."""
    x = rets.to_numpy(dtype=np.float64)
    mv = m.reindex(rets.index).to_numpy(dtype=np.float64)[:, None]
    sx = np.where(np.isfinite(x) & (np.abs(x) > neutral), np.sign(x), 0.0)
    sm = np.where(np.isfinite(mv) & (np.abs(mv) > neutral), np.sign(mv), 0.0)
    both = (sx != 0) & (sm != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        agree = ((sx == sm) & both).sum(axis=0) / both.sum(axis=0)
    return pd.Series(agree, index=rets.columns)


def market_returns(rets: pd.DataFrame, params: MethodParams, freq: str | None) -> pd.Series:
    """Retornos do fator de mercado: ativo informado (ex.: BOVA11) ou média igualitária do universo."""
    if params.market is not None and len(params.market):
        m = to_returns(params.market.astype(float), freq).reindex(rets.index)
        if m.notna().sum() >= 10:
            return m
    return rets.mean(axis=1, skipna=True)


def residualize(rets: pd.DataFrame, m: pd.Series) -> pd.DataFrame:
    """Resíduos de r_i = a_i + b_i·m + e, com a e b estimados em cada ativo (MQO)."""
    x = rets.to_numpy(dtype=np.float64)
    mv = m.to_numpy(dtype=np.float64)[:, None]
    mask = np.isfinite(x) & np.isfinite(mv)
    cnt = mask.sum(axis=0)
    mm = np.where(mask, mv, 0.0)
    xx = np.where(mask, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_m = mm.sum(axis=0) / cnt
        mean_x = xx.sum(axis=0) / cnt
        dm = np.where(mask, mv - mean_m, 0.0)
        beta = (dm * np.where(mask, x - mean_x, 0.0)).sum(axis=0) / (dm * dm).sum(axis=0)
        resid = x - (mean_x + beta * (mv - mean_m))
    resid[~mask] = np.nan
    return pd.DataFrame(resid, index=rets.index, columns=rets.columns)


def corr_matrix(
    rets: pd.DataFrame, method: str, min_obs: int, params: MethodParams, freq: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method == "pearson":
        return pairwise_corr(rets, min_obs)
    if method == "sign":
        return sign_concordance(rets, min_obs, params.neutral)
    if method == "spearman":
        # postos por ativo (aproximação: o exato re-ranqueia em cada par; igual quando não há falhas)
        return pairwise_corr(rets.rank(), min_obs)
    if method == "kendall":
        if rets.shape[1] > KENDALL_MAX_ASSETS:
            raise ValueError(f"Kendall é lento: use no máximo {KENDALL_MAX_ASSETS} ativos "
                             f"(há {rets.shape[1]}). Para universos grandes use Spearman.")
        matrix = rets.corr(method="kendall", min_periods=min_obs)
        valid = rets.notna().astype(np.int64)
        return matrix, valid.T @ valid
    if method == "ewma":
        t = len(rets)
        w = 0.5 ** ((t - 1 - np.arange(t)) / max(params.halflife, 1.0))
        return pairwise_corr(rets, min_obs, weights=w)
    if method == "residual":
        m = market_returns(rets, params, freq)
        res = residualize(rets, m)
        if params.market_name and params.market_name in res:
            res[params.market_name] = np.nan  # o próprio índice não tem resíduo útil
        return pairwise_corr(res, min_obs)
    if method == "stress":
        m = market_returns(rets, params, freq)
        thr = m.quantile(params.tail_q)
        rows = (m <= thr).to_numpy()
        return pairwise_corr(rets, min(min_obs, max(int(rows.sum() * 0.5), 10)), rows=rows)
    raise ValueError(f"Método desconhecido: {method}")


def _issuer_key(code: str) -> str:
    """Chave de 'mesmo ativo': prefixo de 4 letras na B3 (PETR3/PETR4 -> PETR); para externos,
    a série de origem (DOLAR/WDO/DOL -> BRL=X)."""
    from quantbt.external import CATALOG

    a = CATALOG.get(code)
    if a is not None:
        return f"ext:{a.source}:{a.symbol}"
    return code[:4] if len(code) >= 4 else code


def same_issuer(a: str, b: str) -> bool:
    """Mesma empresa/emissor na B3 (PETR3 × PETR4) ou mesma série externa (DOLAR × WDO, IBOV × WIN)."""
    return a != b and _issuer_key(a) == _issuer_key(b)


def analyze(
    prices: pd.DataFrame,
    freq: str | None = None,
    method: str = "pearson",
    min_obs: int = 60,
    rolling_window: int = 60,
    min_coverage: float = 0.5,
    rolling_top: int = DEFAULT_ROLLING_TOP,
    compare: list[str] | None = None,
    params: MethodParams | None = None,
) -> CorrelationResult:
    """Matriz de correlação e ranking de pares.

    method: método principal (ordena o ranking e o mapa de calor).
    compare: outros métodos para mostrar lado a lado (colunas corr_<método>).
    min_obs: mínimo de observações em comum para um par entrar no ranking.
    min_coverage: ativos com dados em menos dessa fração dos pregões da própria vida (da estreia à
        última cotação) são descartados. Cada par usa só o período em que os dois existem (min_obs).
    rolling_window: janela da medida móvel (Direção usa concordância móvel; os demais, Pearson); 0 desliga.
    rolling_top: calcula a correlação móvel só para os N pares de maior |correlação|.
    """
    params = params or MethodParams()
    prices = prices.dropna(how="all")
    # cobertura = fração de pregões com dado DENTRO da vida do ativo (da 1ª à última cotação);
    # assim um ETF lançado no meio do período não é descartado só por ser mais novo
    valid = prices.notna().to_numpy()
    first = valid.argmax(axis=0)
    last = len(prices) - 1 - valid[::-1].argmax(axis=0)
    life = np.where(valid.any(axis=0), last - first + 1, 1)
    coverage = pd.Series(valid.sum(axis=0) / life, index=prices.columns)
    count = pd.Series(valid.sum(axis=0), index=prices.columns)
    # também sai quem não tem observações suficientes para formar nenhum par (viraria linha vazia)
    keep = coverage[(coverage >= min_coverage) & (count > min_obs)].index.tolist()
    dropped = [c for c in prices.columns if c not in keep]
    prices = prices[keep]
    if prices.shape[1] < 2:
        raise ValueError("São necessários pelo menos 2 ativos com dados suficientes.")

    rets = to_returns(prices, freq)
    matrix, overlap = corr_matrix(rets, method, min_obs, params, freq)

    notes: list[str] = []
    extra: dict[str, pd.DataFrame] = {}
    for meth in compare or []:
        if meth == method:
            continue
        if meth == "kendall" and rets.shape[1] > KENDALL_MAX_ASSETS:
            notes.append(f"Kendall omitido na comparação (mais de {KENDALL_MAX_ASSETS} ativos).")
            continue
        extra[meth] = corr_matrix(rets, meth, min_obs, params, freq)[0]

    cols = np.array(matrix.columns, dtype=object)
    iu, ju = np.triu_indices(len(cols), k=1)
    vals = matrix.to_numpy()[iu, ju]
    ok = np.isfinite(vals)
    iu, ju, vals = iu[ok], ju[ok], vals[ok]
    a, b = cols[iu], cols[ju]
    pairs = pd.DataFrame({
        "a": a, "b": b, "corr": vals,
        "obs": overlap.to_numpy()[iu, ju].astype(np.int64),
    })
    keys = np.array([_issuer_key(c) for c in cols], dtype=object)  # uma vez por ativo, não por par
    pairs["same_issuer"] = keys[iu] == keys[ju]
    for meth, mat in extra.items():
        pairs[f"corr_{meth}"] = mat.to_numpy()[iu, ju]
    if method == "sign" or "sign" in extra:
        agree = market_agreement(rets, market_returns(rets, params, freq), params.neutral)
        pairs["mkt_a"] = agree.reindex(a).to_numpy()
        pairs["mkt_b"] = agree.reindex(b).to_numpy()

    if len(pairs) and rolling_window and rolling_window >= 5:
        for k in ("roll_mean", "roll_std", "roll_min", "roll_last"):
            pairs[k] = np.nan
        top = pairs["corr"].abs().nlargest(rolling_top).index if rolling_top else pairs.index
        stats = []
        for i in top:
            rc = rolling_measure(method, rets, pairs.at[i, "a"], pairs.at[i, "b"], rolling_window, params.neutral)
            stats.append((rc.mean(), rc.std(), rc.min(), rc.iloc[-1] if len(rc) else np.nan))
        if stats:
            pairs.loc[top, ["roll_mean", "roll_std", "roll_min", "roll_last"]] = np.array(stats, dtype=float)

    pairs = pairs.sort_values("corr", ascending=False, ignore_index=True)
    pairs.insert(0, "rank", np.arange(1, len(pairs) + 1))
    return CorrelationResult(prices=prices, returns=rets, matrix=matrix, pairs=pairs, dropped=dropped,
                             method=method, compared=list(extra), notes=notes, params=params,
                             window=rolling_window)


def rolling_corr(returns: pd.DataFrame, a: str, b: str, window: int = 60) -> pd.Series:
    pair = returns[[a, b]].dropna()
    return pair[a].rolling(window, min_periods=window).corr(pair[b]).dropna()


def rolling_sign(returns: pd.DataFrame, a: str, b: str, window: int = 60, neutral: float = 0.0) -> pd.Series:
    """Concordância de direção móvel (mesma escala de -1 a 1 do método 'sign')."""
    pair = returns[[a, b]].dropna()
    s = np.sign(pair.where(pair.abs() > neutral, 0.0))
    prod = s[a] * s[b]
    moved = (prod != 0).astype(float)
    num = prod.rolling(window, min_periods=window).sum()
    den = moved.rolling(window, min_periods=window).sum()
    return (num / den.replace(0, np.nan)).dropna()


def rolling_measure(res_method: str, returns: pd.DataFrame, a: str, b: str, window: int, neutral: float = 0.0):
    if res_method == "sign":
        return rolling_sign(returns, a, b, window, neutral)
    return rolling_corr(returns, a, b, window)


def cluster_order(matrix: pd.DataFrame) -> list[str]:
    """Ordena os ativos para agrupar os correlacionados no mapa de calor (vizinho mais próximo)."""
    m = matrix.fillna(0).to_numpy(copy=True)
    names = matrix.columns.tolist()
    n = len(names)
    if n <= 2:
        return names
    np.fill_diagonal(m, -np.inf)
    used = np.zeros(n, dtype=bool)
    cur = int(np.argmax(np.abs(np.where(np.isfinite(m), m, 0)).sum(axis=0)))
    order = [cur]
    used[cur] = True
    for _ in range(n - 1):
        row = np.where(used, -np.inf, m[cur])
        cur = int(np.argmax(row))
        order.append(cur)
        used[cur] = True
    return [names[i] for i in order]
