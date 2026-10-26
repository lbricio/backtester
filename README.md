# quantbt — Backtester de estratégias quantitativas

Aplicativo desktop (Qt/PySide6-Essentials) para Linux e macOS.

**Guia de uso passo a passo: [COMO_USAR.md](COMO_USAR.md).**

## Instalação

Requer [uv](https://docs.astral.sh/uv/) (ou Python 3.10+ com pip).

```bash
uv sync --extra yahoo     # instala dependências (+ yfinance para baixar cotações)
uv run quantbt            # abre o aplicativo
uv run pytest             # roda os testes
```

No Linux, se o Qt reclamar do plugin "xcb", instale: `sudo apt install libxcb-cursor0`.

## Recursos

- **Dados**: base local da B3 (séries históricas oficiais COTAHIST), série sintética de exemplo, arquivo CSV (detecta `,`/`;` e nomes de colunas em
  português ou inglês) ou Yahoo Finance (ex.: `PETR4.SA`, `^BVSP`, `AAPL`, `BTC-USD`).
- **36 estratégias prontas**: tendência (médias, MACD, Donchian, Bollinger, Keltner, Ichimoku, ADX, Aroon,
  Supertrend, SAR, Heikin-Ashi, setups 9.1 e 123…), reversão à média (RSI, Connors, Bollinger, estocástico,
  Williams %R, CCI…), gestão de risco (alvo de volatilidade, Faber, Turtle), sazonalidade e buy & hold.
- **Editor de estratégia**: escreva sua própria estratégia em Python dentro do app.
- **Custos**: comissão percentual, taxa fixa por ordem e slippage; permite ou não vendido.
- **Taxa livre de risco** (ex.: CDI) para Sharpe e Sortino. Com 0 eles medem o retorno bruto, o que
  no Brasil infla o resultado de qualquer estratégia.
- **Resultados**: gráfico de preço com entradas/saídas, patrimônio vs. buy & hold, drawdown,
  métricas (CAGR, Sharpe, Sortino, Calmar, drawdown, taxa de acerto, profit factor…) e lista de
  operações exportável em CSV.
- **Otimização**: busca em grade dos parâmetros, em segundo plano, com ranking por métrica.
- **Varredura**: várias estratégias × vários ativos de uma vez, com mapa de calor, ranking por robustez
  (% de ativos em que vence o CDI histórico do período e o Buy & Hold) e clique para abrir o par.
- **Simulação**: Monte Carlo do preço futuro (normal, t-Student, bootstrap histórico e em blocos) com
  animação, VaR/CVaR e chance de atingir um alvo; preço justo de opções (Monte Carlo × Black-Scholes,
  gregas, volatilidade implícita); valuation de ações (Graham, Bazin, Gordon, fluxo de caixa descontado
  com incerteza por Monte Carlo). O CDI atual vem do Banco Central.
- **Correlação entre ativos**: carrega dezenas de ativos de uma vez, gera a matriz de correlação
  (mapa de calor agrupado), ranqueia os pares e mostra a correlação móvel de cada par.

## Dados históricos da B3

Na aba **Base B3**, escolha os anos e clique em *Baixar anos*. O app baixa os arquivos anuais
`COTAHIST_A{ANO}.ZIP` do site da B3, mantém só o mercado à vista (ações, units, FIIs, Fiagro, ETFs, BDRs, inclusive empresas em
recuperação judicial; descarta direitos, recibos e bônus de subscrição)
e salva em Parquet em `~/.quantbt/b3/` (~5–9 MB por ano) (só precisa baixar uma vez; use *Atualizar ano atual*
para os pregões novos). Se o download direto falhar, baixe os arquivos pelo site da B3 e use
*Importar COTAHIST*.

Os preços da B3 **não são ajustados por proventos**. O app corrige automaticamente
desdobramentos e grupamentos (saltos com razões típicas como 2:1 ou 1:10), mas não desconta
dividendos. Para preços totalmente ajustados, use a fonte Yahoo Finance.

Na leitura, o app também:
- descarta **direitos e recibos de subscrição** (ex.: ALZR12, BPAC13), que duram poucos dias;
- une pelo **ISIN** papéis que trocaram de código (ex.: `PET 4` → PETR4, antes de 1998);
- classifica cada ativo pela categoria do **pregão mais recente** (em 10/10/2022 a B3 moveu todos os
  BDRs da categoria de ações para a de BDRs);
- separa **códigos reaproveitados** por outro papel: o mais recente fica com o código e o antigo vira,
  por exemplo, `FIXX11~2019` (FII encerrado em 2019; o ETF FIXX11 atual começou em 2025).

No backtest, escolha a fonte *Base B3* e digite o código (ex.: `PETR4`).

## Dólar, mini dólar, commodities, cripto e índices

Ativos fora da B3 têm apelidos curtos que funcionam na **Correlação** (misturados com ações) e no
**Backtest** (fonte *Base B3*, digite o apelido). Na Correlação, use **Externos → + Adicionar** para
incluir grupos prontos. Os dados (10 anos) vêm do Yahoo Finance e do Banco Central (PTAX) e ficam em
cache em `~/.quantbt/externos/`, atualizados automaticamente após 12 h.

| Grupo | Apelidos |
|---|---|
| Dólar e moedas | `DOLAR` (USD/BRL), `PTAX` (Banco Central), `EURO`, `DXY`, `YUAN` |
| Futuros B3 (substitutos) | `WDO`, `DOL` → dólar à vista · `WIN`, `IND` → Ibovespa |
| Commodities | `PETROLEO` (WTI), `BRENT`, `GAS`, `OURO`, `PRATA`, `PLATINA`, `COBRE`, `ALUMINIO`, `MINERIO`, `SOJA`, `MILHO`, `TRIGO`, `CAFE`, `ACUCAR`, `ALGODAO`, `CACAU`, `BOI` |
| Cripto | `BTC`, `ETH` (US$), `BTCBRL`, `ETHBRL` (R$) |
| Índices | `IBOV`, `SP500`, `NASDAQ`, `VIX`, `US10Y` |

Cuidados:
- **Futuros da B3** (WDO, DOL, WIN, IND) não estão nos arquivos COTAHIST. Os apelidos usam o dólar à
  vista e o Ibovespa, que diferem do futuro só pelo custo de carregamento (juros). Por isso DOLAR × WDO
  dá 1,00 e o par é tratado como "mesmo ativo" (escondido por *Ignorar mesma empresa*).
- **Horários de fechamento** diferentes (commodities e S&P nos EUA, PTAX às 13h, cripto 24 h)
  subestimam a correlação **diária** com a B3. Para comparar mercados, prefira retornos **semanais**.
  Para correlação com a B3, `DOLAR` é mais sincronizado que `PTAX`.
- O câmbio do Yahoo vem datado um dia à frente; o app corrige (verificado contra PTAX e Ibovespa).
- Commodities são futuros contínuos (1º vencimento), em US$. *Externos em R$* converte pelo dólar.
- Cripto negocia 7 dias por semana; as métricas anualizadas usam ~365 períodos/ano.

```bash
uv run quantbt-cli externos --list          # catálogo
uv run quantbt-cli externos --update        # baixa/atualiza tudo
uv run quantbt-cli corr --kind acoes --max-assets 40 --preset tudo --freq Semanal
uv run quantbt-cli corr --kind acoes --extra DOLAR BTC OURO PETROLEO --method sign
```

## Correlação entre ativos

1. Na aba **Correlação**, digite os códigos ou use **Universo da base B3** → *Carregar universo*:
   - **Tipo**: todos, ações/units, BDRs, FIIs/Fiagro ou ETFs/fundos;
   - **Volume mín.**: volume financeiro médio por pregão no período (ex.: 500 mil);
   - **Negociado em ≥**: fração mínima dos pregões em que o ativo negociou (ex.: 90%);
   - **Máx. de ativos**: limita aos N mais líquidos ("todos" = sem limite).
2. Escolha período, frequência dos retornos (diária, semanal, mensal) e método (veja abaixo).
3. *Calcular correlações* gera o mapa de calor e o ranking dos pares. Ordene por mais
   correlacionados, mais negativos, maior |correlação|, mais próximos de zero ou mais estáveis.
   *Ignorar mesma empresa* esconde pares óbvios como PETR3 × PETR4.
4. Clique num par (na tabela ou no mapa) para ver os preços normalizados e a correlação móvel.

### Métodos de correlação

Todos usam retornos logarítmicos, r = ln(Pₜ / Pₜ₋₁), apenas nos pregões em que os dois ativos
negociaram. *Comparar métodos* mostra todos lado a lado no ranking.

| Método | Fórmula / ideia | Para quê |
|---|---|---|
| **Direção (sinais)** | (dias na mesma direção − dias opostos) ÷ dias com movimento dos dois | Reação conjunta, ignorando a intensidade. % de dias juntos = (1 + valor) ÷ 2. *Zona neutra* ignora dias com variação pequena |
| **Pearson** | (nΣxy − ΣxΣy) / √[(nΣx² − (Σx)²)(nΣy² − (Σy)²)] | Relação linear; padrão do mercado; sensível a dias extremos |
| **Spearman** | Pearson sobre os postos (rankings) dos retornos | Robusto a outliers |
| **Kendall** | (pares de dias concordantes − discordantes) ÷ total de pares | O mais robusto; lento (até 80 ativos) |
| **Exponencial (EWMA)** | Pearson com peso 0,5^(idade/meia-vida) | Relação atual, com mais peso no recente |
| **Residual** | Pearson dos resíduos de r = a + β·mercado + ε | Tira o efeito do Ibovespa: mostra relações próprias (setor, holding) |
| **Estresse** | Pearson só nos X% piores dias do mercado | Se a diversificação some nas quedas |

O ranking também mostra **A × mercado** e **B × mercado**: % dos dias em que cada ativo foi na
mesma direção do fator de mercado (padrão BOVA11; vazio = média do universo).

Ativos ilíquidos distorcem o resultado, por isso o filtro de liquidez é importante. A medida móvel
(Pearson, ou concordância de direção no método Direção) é calculada para os 3.000 pares mais fortes.

### Universo inteiro pela linha de comando

```bash
uv run quantbt-cli b3 --from 1986 --to 2026      # baixa o histórico completo (~40 anos)
uv run quantbt-cli b3 --update                   # atualiza o ano corrente
uv run quantbt-cli corr --start 2023-01-01 --min-volume 1000000 --out resultados/
uv run quantbt-cli corr --kind acoes fiis --freq Semanal --show 50
uv run quantbt-cli corr --kind acoes --method sign --compare   # direção + todos os métodos
```

Gera `ranking_pares.csv` (todos os pares, com nome e tipo de cada ativo), `matriz_correlacao.csv`
e `universo.csv`.

## Modelo de execução

A estratégia devolve, para cada candle, a posição alvo como fração do patrimônio
(`1` comprado, `0` fora, `-1` vendido). O sinal calculado no **fechamento** do candle *t* é
executado na **abertura** do candle *t+1*, evitando look-ahead. Só há ordem quando o alvo muda.

## Estratégia personalizada

```python
NAME = "Minha estratégia"
PARAMS = {"janela": {"default": 50, "min": 10, "max": 200, "step": 10}}

def generate_signals(df, janela):
    return (df["close"] > sma(df["close"], janela)).astype(float)
```

Disponíveis no editor: `pd`, `np`, `sma`, `ema`, `rsi`, `atr`. Arquivos salvos ficam em
`~/.quantbt/strategies/`.

## Estrutura

```
quantbt/
  engine.py       motor de backtest (execução, custos, operações)
  metrics.py      métricas de performance
  strategies.py   estratégias prontas, indicadores e estratégia por código
  optimizer.py    busca em grade
  data.py         CSV, Yahoo Finance e dados sintéticos
  b3.py           download, leitura e base local das séries históricas da B3
  correlation.py  matriz e ranking de correlação
  montecarlo.py   simulação do preço futuro
  options.py      Black-Scholes, Monte Carlo de opções, gregas, vol. implícita
  valuation.py    Graham, Bazin, Gordon, fluxo de caixa descontado
  sources.py      carregamento unificado de preços e CDI
  scan.py         varredura estratégias × ativos
  external.py     dólar, commodities, cripto e índices (Yahoo/Banco Central)
  cli.py          linha de comando (download da B3, externos, correlação do universo)
  gui/            interface (janela principal, gráficos, editor)
tests/            testes
```
