# Como usar o quantbt

Guia prático do app: dados, backtest, estratégias, otimização, correlação, simulação (Monte Carlo,
opções e valuation) e linha de comando.
Para instalação e detalhes técnicos, veja o [README](README.md).

---

## 1. Abrir o app

```bash
cd ~/projects/quantbt
~/.local/bin/uv run quantbt
```

A janela tem cinco **seções** na barra de cima:

| Seção | Para quê |
|---|---|
| **Backtest** | Testar uma estratégia num ativo |
| **Correlação** | Comparar muitos ativos e ranquear quem anda junto |
| **Simulação** | Monte Carlo do preço futuro, preço justo de opções e valuation de ações |
| **Base B3** | Baixar e atualizar os dados |
| **Log** | Histórico do que o app fez, incluindo erros |

A seção **Backtest** tem sub-abas próprias: **Gráficos**, **Métricas**, **Operações**,
**Editor de estratégia** e **Otimização**. O painel da esquerda (dados, estratégia, custos) vale
para todas elas e só aparece nessa seção.

---

## 2. Os dados

### Base B3 (principal)
Arquivos oficiais da B3 (COTAHIST) com todos os ativos do mercado à vista: ações, units, FIIs,
Fiagro, ETFs e BDRs, de 1986 até hoje. Ficam em `~/.quantbt/b3/`.

- **Atualizar os pregões novos**: seção **Base B3 → "Atualizar <ano atual>"**. Faça isso de vez em quando.
- Baixar anos que faltam: escolha o intervalo e clique **Baixar só os que faltam**.
- Se o download falhar (rede corporativa, proxy), baixe os arquivos no site da B3 e use
  **Importar COTAHIST**.

O app corrige automaticamente:
- desdobramentos, grupamentos e bonificações de 50%;
- trocas de código (ex.: `PET 4` → PETR4), unindo pelo ISIN;
- códigos reaproveitados por outro papel (o antigo vira, por exemplo, `FIXX11~2019`);
- direitos e recibos de subscrição (descartados);
- moedas antigas (preços antes de 1994 convertidos para reais).

> ⚠️ **Dividendos não são descontados.** Ações que pagam muito provento (PETR4, bancos) parecem
> render menos do que renderam de fato. Para preços totalmente ajustados, use a fonte Yahoo Finance.

### Externos: dólar, commodities, cripto e índices
Vêm do Yahoo Finance e do Banco Central (PTAX), com 10 anos de histórico. Ficam em
`~/.quantbt/externos/` e se atualizam sozinhos quando têm mais de 12 h (sem internet, o app usa o
que já tem). Use pelo apelido:

| Grupo | Apelidos |
|---|---|
| Dólar e moedas | `DOLAR`, `PTAX`, `EURO`, `DXY`, `YUAN` |
| Futuros B3 (substitutos) | `WDO`, `DOL` → dólar à vista · `WIN`, `IND` → Ibovespa |
| Commodities | `PETROLEO`, `BRENT`, `GAS`, `OURO`, `PRATA`, `PLATINA`, `COBRE`, `ALUMINIO`, `MINERIO`, `SOJA`, `MILHO`, `TRIGO`, `CAFE`, `ACUCAR`, `ALGODAO`, `CACAU`, `BOI` |
| Cripto | `BTC`, `ETH` (US$), `BTCBRL`, `ETHBRL` (R$) |
| Índices | `IBOV`, `SP500`, `NASDAQ`, `VIX`, `US10Y` |

> Os futuros da B3 (WDO, DOL, WIN, IND) não estão nos arquivos da B3. Os apelidos usam o dólar à
> vista e o Ibovespa, que diferem do futuro só pelo custo de carregamento (juros).

Para atualizar tudo de uma vez: **Base B3 → "Atualizar dados externos"**.

---

## 3. Backtest: testar uma estratégia

### Passo a passo
1. **Dados** (painel esquerdo): Fonte = *Base B3*, Ticker = `PETR4` (ou `BTC`, `DOLAR`, `OURO`…),
   período → **Carregar dados**. O campo de ticker sugere os apelidos enquanto você digita.
2. **Estratégia**: escolha uma da lista e ajuste os parâmetros.
3. **Execução e custos**:
   - **Comissão** e **slippage**: uns 0,05% cada é razoável;
   - **Livre de risco**: coloque o CDI atual (ex.: 10,5). Sem isso, o Sharpe fica inflado;
   - **Permitir posições vendidas**: se a estratégia pode apostar na queda.
4. **▶ Rodar backtest** (ou **Ctrl+R**).

Outras fontes: *Yahoo Finance* (ex.: `PETR4.SA`, `AAPL`, `BTC-USD`, com intervalos intradiários),
*Arquivo CSV* (detecta `,`/`;` e nomes de colunas em português ou inglês) e *Dados de exemplo*.

### Estratégias prontas (36)
**Seguimento de tendência** (ganham em movimentos longos; sofrem em mercado lateral):
| Estratégia | Ideia |
|---|---|
| Cruzamento de médias | Comprado quando a média curta está acima da longa |
| Médias exponenciais | O mesmo, com médias exponenciais (reagem mais rápido) |
| Cruzamento triplo de médias | Comprado com as três médias alinhadas para cima |
| Preço cruza a média | Comprado com o preço acima da média de N dias |
| Acima da média de 200 | O clássico filtro de tendência de 200 dias |
| MACD | Comprado com a linha MACD acima da linha de sinal |
| Rompimento de Donchian | Compra na máxima de N dias; sai na mínima de M dias |
| Rompimento de Bollinger | Compra acima da banda superior; sai na média |
| Rompimento de Keltner | Compra acima do canal de Keltner (média + k × ATR) |
| Máxima de 52 semanas | Compra na máxima de 1 ano; sai X% abaixo dela |
| ADX / DMI | Comprado com tendência forte e de alta |
| Aroon | Comprado quando as máximas são mais recentes que as mínimas |
| Ichimoku (nuvem) | Comprado acima da nuvem com Tenkan > Kijun |
| Supertrend | Faixa baseada no ATR; troca de lado no rompimento |
| Parabolic SAR | Stop móvel que acelera com a tendência |
| Saída Chandelier | Entra acima da média; sai no stop "máxima − k × ATR" |
| Heikin-Ashi | Comprado após N candles Heikin-Ashi de alta |
| Setup 9.1 (Larry Williams) | Compra quando a média exponencial de 9 vira para cima |
| Setup 123 de compra | Padrão de fundo em três candles; compra no rompimento |
| IFR cruzando 50 | Comprado com o RSI acima de 50 |
| OBV (volume) | Comprado com o volume acumulado (OBV) acima da média |
| Momentum | Comprado se subiu nos últimos N dias |

**Reversão à média** (compram quedas exageradas; sofrem em quedas que viram tendência):
| Estratégia | Ideia |
|---|---|
| Reversão por RSI | Compra com RSI muito baixo; sai quando ele se recupera |
| RSI(2) de Connors | Compra quedas fortes de curto prazo, só em tendência de alta |
| Bandas de Bollinger | Compra abaixo da banda inferior; sai na média |
| Envelope de médias | Compra X% abaixo da média; sai na média |
| Estocástico | Compra ao sair da sobrevenda; vende na sobrecompra |
| Williams %R | O mesmo, com o %R (−80 e −20) |
| CCI | Compra quando o CCI volta para cima de −100 |
| Quedas consecutivas | Compra após N quedas seguidas; sai na primeira alta |

**Gestão de risco e alocação** (usadas por gestoras):
| Estratégia | Ideia |
|---|---|
| Tendência com alvo de volatilidade (CTA) | Momentum de 12 meses com posição ajustada à volatilidade |
| Buy & Hold com alvo de volatilidade | Sempre comprado, reduzindo a posição quando a volatilidade sobe |
| Timing de 10 meses (Faber) | Comprado se o preço está acima da média de 10 meses |
| Tartarugas (Turtle) com risco por ATR | Rompimento com posição dimensionada pelo risco |

**Sazonalidade e referência:**
| Estratégia | Ideia |
|---|---|
| Virada do mês | Comprado nos últimos e primeiros pregões de cada mês |
| Buy & Hold | Compra e segura (referência) |

As que têm o parâmetro "Operar vendido (0/1)" ficam vendidas na baixa com 1 (precisa de "Permitir
posições vendidas" ligado). Todas são testadas para nunca usar dados do futuro.

### Como o app simula
- A decisão é tomada com o **fechamento** do dia e executada na **abertura do dia seguinte**.
  Assim não há como "ver o futuro".
- Só há ordem quando a posição desejada muda; comissão e slippage são cobrados em cada ordem.

### Lendo os resultados
- **Gráficos**: preço com ▲ compras e ▼ vendas, patrimônio comparado ao Buy & Hold e drawdown
  (quanto caiu desde o pico).
- **Métricas**, comparando estratégia e Buy & Hold:
  - **CAGR**: retorno médio por ano;
  - **Sharpe**: retorno acima do CDI por unidade de risco. Acima de 1 é bom; negativo significa que
    perde para o CDI;
  - **Drawdown máximo**: a pior queda, o que você teria que aguentar;
  - **Taxa de acerto** e **profit factor** (ganhos ÷ perdas; acima de 1 é lucrativo).
- **Operações**: cada trade com entrada, saída e resultado. **Exportar operações (CSV)**.
- Menu **Arquivo → Exportar curva de patrimônio** salva o patrimônio dia a dia.

---

## 4. Sua própria estratégia (sub-aba Editor de estratégia)

Escreva em Python. O modelo já vem pronto (botão **Restaurar modelo**):

```python
NAME = "Minha estratégia"
PARAMS = {"janela": {"default": 50, "min": 10, "max": 200, "step": 10}}

def generate_signals(df, janela):
    # 1 = comprado, 0 = fora, -1 = vendido (0.5 = meia posição)
    return (df["close"] > sma(df["close"], janela)).astype(float)
```

- `df` tem as colunas `open`, `high`, `low`, `close` e `volume` (índice = datas).
- Prontos para uso: `sma`, `ema`, `rsi`, `atr`, `macd`, `stochastic`, `adx`, `supertrend`,
  `parabolic_sar`, além de `pd` (pandas) e `np` (numpy).
- **Aplicar e rodar** testa na hora. Erros de código aparecem numa mensagem, sem travar o app.
- **Salvar** grava em `~/.quantbt/strategies/`; **Abrir** carrega de lá.
- Os parâmetros de `PARAMS` aparecem no painel e podem ser otimizados.

---

## 5. Otimização (sub-aba Otimização)

1. Marque os parâmetros a variar e defina mínimo, máximo e passo (o app mostra quantas combinações
   dá; o limite é 20 mil).
2. Escolha a métrica (Sharpe, CAGR, Calmar…) → **▶ Otimizar** (ou **Ctrl+Shift+R**). **Parar**
   interrompe.
3. **Dois cliques** num resultado aplica os parâmetros e roda o backtest.

> ⚠️ **Sobreajuste:** o melhor resultado foi escolhido olhando o mesmo período em que é medido, então
> tende a ser otimista. Desconfie de parâmetros "perfeitos" e prefira regiões em que vários vizinhos
> também vão bem.

---

## 5.1 Varredura: várias estratégias em vários ativos (sub-aba Varredura)

Testa todas as combinações de estratégias × ativos de uma vez, com os custos, o slippage, a regra de
vendido e a taxa livre de risco do painel lateral.

1. **Ativos**: digite os códigos, ou use **Carregar universo da B3** (tipo, máximo de ativos, volume
   mínimo, % de pregões, incluir estreantes) e **+ Externos** (BTC, dólar, commodities...).
2. **Estratégias**: marque as que quer testar (**Todas** / **Nenhuma**). Cada uma usa os parâmetros
   padrão; a que está selecionada no painel usa os valores do painel. Estratégias aplicadas no Editor
   entram na lista sozinhas.
3. **Período** (padrão: 10 anos) → **▶ Rodar varredura**. **Parar** interrompe e mostra o que já rodou.

**Resultados:**
- **Mapa de calor**: ativo × estratégia. Escolha a medida em *Mapa*: ganho sobre o CDI, ganho sobre o
  Buy & Hold, Sharpe, CAGR ou drawdown. Azul é bom, vermelho é ruim; clique numa célula para abrir o par
  no backtest.
- **Ranking de estratégias**: em quantos % dos ativos cada uma **vence o CDI** e o Buy & Hold, e os
  valores **medianos** de ganho, Sharpe e drawdown.
- **Ranking de ativos**: melhor estratégia de cada ativo e quantas estratégias vencem o CDI nele.
- **Todos os resultados**: cada combinação, com erros (ex.: histórico curto). Dois cliques abrem no
  backtest. **Exportar resultados** salva em CSV.

**Como ler:** o CDI de comparação é o que **realmente rendeu nas mesmas datas** de cada backtest
(Banco Central). Nos últimos 10 anos foi cerca de 9,4% a.a., bem diferente do CDI de hoje. Diferenças
menores que 0,1 ponto por ano contam como empate. Prefira estratégias que vencem em **muitos ativos**
com **mediana positiva**: um único resultado brilhante costuma ser sorte.

---

## 6. Correlação: quem anda junto

### Passo a passo
1. **Ativos**: digite os códigos (espaço, vírgula ou linha) ou monte um universo:
   - **Universo da base B3**: tipo (ações, FIIs, BDRs, ETFs), volume mínimo por dia e
     "dias c/ negócio ≥ 90%" (o ativo precisa ter negociado em pelo menos 90% dos pregões; descarta
     ilíquidos) → **Carregar universo**. Com **"Incluir quem estreou
     depois do início"** ligado (padrão), esses filtros contam a partir da estreia de cada ativo, então
     entram IPOs e ETFs recentes (ex.: BBOV11, de 2020), desde que tenham o **histórico mínimo** (padrão:
     1 ano). Desligado, só entram ativos negociados durante todo o período;
   - **Externos**: escolha um grupo (Commodities, Cripto, Dólar…) → **+ Adicionar**.
     "Externos em R$" converte o que é cotado em dólar.
2. **Período e método**: período (padrão: 10 anos), retornos **diários**, **semanais** ou **mensais**
   e o método.
3. **▶ Calcular correlações**.

### Qual método usar
| Pergunta | Método |
|---|---|
| "Se um sobe, o outro sobe?" (só a direção) | **Direção (sinais)**, padrão |
| Relação clássica, com intensidade | **Pearson** |
| Igual, mas sem se deixar levar por dias extremos | **Spearman** |
| O mais robusto (lento, até 80 ativos) | **Kendall** |
| Como está a relação *hoje* | **Exponencial (EWMA)** |
| Quem anda junto *tirando o efeito do Ibovespa* | **Residual** |
| A diversificação some nas crises? | **Estresse** |

Com **"Comparar todos os métodos no ranking"** ligado, todos aparecem lado a lado. Um par que fica
forte em todos os métodos tem uma relação sólida.

Opções dos métodos:
- **Meia-vida** (EWMA): idade em que o dado passa a valer metade (126 pregões ≈ 6 meses);
- **Mercado**: fator usado no Residual, no Estresse e em "× mercado" (padrão BOVA11);
- **Estresse**: fração dos piores dias do mercado (padrão 20%);
- **Zona neutra** (Direção): dias com variação menor que isso contam como "parado".

### Lendo a tela
- **Mapa de calor**: azul = andam juntos, vermelho = opostos. Passe o mouse para ver o valor e
  clique para abrir o par.
- **Ranking**: ordene por mais correlacionados, mais negativos, maior |correlação|, mais próximos de
  zero ou mais estáveis. O **filtro** procura um ativo. **Ignorar mesma empresa** esconde pares
  óbvios (PETR3 × PETR4, DOLAR × WDO).
- **Direção**: +0,6 significa 80% dos dias na mesma direção, porque (1 + valor) ÷ 2 = 0,8.
- **A × mercado / B × mercado**: % dos dias em que cada ativo seguiu o fator de mercado. É a
  medida de reatividade ao mercado.
- **Média, desvio, mínima e atual (móvel)**: como a relação variou ao longo do tempo. Desvio
  baixo = relação estável.
- **Obs.**: quantos períodos o par tem em comum. Pares com ativos recentes usam só o período em que os
  dois existem; desconfie de pares com poucas observações.
- **Gráfico do par** (embaixo): preços lado a lado e a relação ao longo do tempo.
- **Exportar ranking / matriz** salva em CSV. **Abrir ativo A no backtest** leva o ativo direto
  para testar estratégias.

### Cuidados
- **Ao misturar B3 com externos, use retornos semanais.** Commodities, S&P e cripto fecham em outros
  horários, e no diário isso esconde parte da correlação (o app avisa).
- **Correlação não é causa nem garantia.** Uma relação forte nos últimos 10 anos pode mudar.
- **Correlação não é cointegração**: dois ativos podem andar juntos no dia a dia e se afastar no
  longo prazo. Para operar pares (long & short), isso precisa ser testado à parte.

---

## 7. Simulação: Monte Carlo e preço justo

A seção **Simulação** tem três sub-abas. Nas três, a taxa de juros já vem preenchida com o **CDI
atual**, buscado no Banco Central ao abrir o app.

### 7.1 Monte Carlo (preço futuro)
Simula milhares de futuros possíveis para o preço de um ativo. O gráfico vai se formando enquanto as
simulações rodam.

1. **Ativo**: código da B3 ou externo (`PETR4`, `BTC`, `DOLAR`…) e quantos anos de **histórico**
   usar para medir a volatilidade (padrão: 2).
2. **Modelo**:
   | Modelo | Ideia |
   |---|---|
   | Normal (GBM) | Clássico; subestima quedas extremas |
   | Caudas gordas (t-Student) | Mesma volatilidade, mas dias extremos mais frequentes |
   | Histórico (bootstrap) | Sorteia retornos diários que aconteceram de fato |
   | Histórico em blocos | Sorteia blocos de dias seguidos (mantém fases calmas e turbulentas) |
3. **Tendência** (quanto o preço sobe em média):
   - *Taxa livre de risco* (padrão): rende em média o CDI;
   - *Histórica*: repete a tendência do período. **Cuidado:** com 2 anos de histórico, a margem de erro
     dessa média é enorme (cerca de ±20 pontos por ano);
   - *Sem tendência*: preço esperado igual ao atual;
   - *Personalizada*: o retorno anual que você informar.
4. **Volatilidade**: "histórica" ou um valor seu (aumente para simular um cenário de estresse).
5. **Horizonte** (252 pregões ≈ 1 ano), **simulações** (até 50 mil), **preço-alvo** (opcional) e
   **semente** (um número fixo repete os mesmos sorteios).
6. **▶ Simular**. **Pausar**, **Parar** e **Velocidade** controlam a animação.

**Como ler:** cada linha azul é um futuro possível. A faixa laranja clara contém 90% das trajetórias e
a escura 50%; a linha laranja é a mediana. À direita, a distribuição do preço no fim. A tabela mostra:
- mediana, média e faixas de preço no fim;
- **chance de terminar acima do preço atual** e, com alvo, de **terminar além do alvo** e de
  **tocar o alvo** em algum momento;
- **VaR 95%** (a perda no pior 5% dos cenários) e **CVaR 95%** (a perda média nesses cenários);
- **queda máxima típica** no caminho.

**Exportar percentis** salva as curvas P5, P25, P50, P75 e P95 por data.

> A simulação supõe que o futuro terá a volatilidade do período escolhido. Ela mostra a faixa de
> incerteza, não uma previsão.

### 7.2 Opções (preço justo)
Calcula quanto vale uma call ou put europeia por **Monte Carlo** e compara com **Black-Scholes**.

1. Informe o ativo e clique **Usar preço e volatilidade do ativo** (preenche S, o strike no dinheiro e
   a volatilidade histórica).
2. Ajuste **tipo** (compra/venda), **strike** e **vencimento** (o prazo em dias úteis é calculado
   sozinho; feriados não são descontados).
3. **Juros** (CDI), **volatilidade** e **dividendos** esperados até o vencimento.
4. Opcional: **preço de mercado** da opção, para calcular a **volatilidade implícita**.
5. **▶ Calcular preço justo**.

**Como ler:**
- o gráfico da esquerda mostra a estimativa do Monte Carlo **convergindo** para o Black-Scholes
  (linha tracejada), com a faixa de confiança de 95% se estreitando;
- à direita, o preço do ativo no vencimento simulado; em verde, onde a opção termina dentro do dinheiro;
- a tabela traz o preço justo pelos dois métodos, a **chance de exercício** e as gregas: **delta**
  (por R$ 1 no ativo), **gamma**, **vega** (por 1 p.p. de volatilidade), **theta** (por dia útil) e
  **rho** (por 1 p.p. de juros);
- com o preço de mercado: se a **volatilidade implícita** for maior que a histórica, a opção está
  "cara" em relação ao que o ativo costuma oscilar; se for menor, "barata".

> Na B3, calls sobre ações costumam ser americanas. Sem dividendos até o vencimento, valem o mesmo que
> a europeia; a put americana pode valer um pouco mais que o calculado.

### 7.3 Valuation (preço justo de ações)
1. Digite a ação e clique **Buscar dados**: preço (da Base B3), LPA, VPA, dividendos (12 meses e
   média de 5 anos) e fluxo de caixa livre por ação vêm do Yahoo. **Todos os campos são editáveis**:
   confira no RI da empresa.
2. Ajuste as **premissas**: multiplicador de Graham (22,5), yield mínimo de Bazin (6%), **desconto**
   (o botão *CDI + 4%* usa o CDI atual mais um prêmio de risco), crescimento nos primeiros anos e
   crescimento perpétuo.
3. **▶ Calcular preço justo**.

| Método | Fórmula | Para quê |
|---|---|---|
| Graham | √(22,5 × LPA × VPA) | Empresas lucrativas com patrimônio positivo |
| Bazin | Dividendos ÷ yield mínimo | Preço teto para quem busca dividendos |
| Gordon | D × (1 + g) ÷ (k − g) | Empresas maduras pagadoras de dividendos |
| Fluxo de caixa descontado | FCF crescendo por N anos + perpetuidade, descontados a k | O mais completo e o mais sensível às premissas |

**Como ler:** a tabela mostra o preço justo de cada método e quanto ele está acima (verde, a ação
parece barata) ou abaixo (vermelho) do preço atual. O gráfico da direita é o **fluxo de caixa
descontado com incerteza**: o app sorteia milhares de combinações de crescimento e desconto (dentro dos
desvios que você definir) e mostra a faixa de preços justos possíveis e em quantos cenários ela fica
acima do preço atual.

> Preço justo depende muito das premissas. Pequenas mudanças no desconto ou no crescimento perpétuo
> mudam bastante o resultado; use as faixas como ordem de grandeza, não como número exato.

---

## 8. Linha de comando (sem abrir a janela)

Útil para rodar o universo inteiro e exportar CSVs:

```bash
uv run quantbt-cli b3 --update                    # atualiza a base da B3 (ano atual)
uv run quantbt-cli b3 --from 2011 --missing       # baixa só os anos que faltam
uv run quantbt-cli externos --list                # catálogo de externos
uv run quantbt-cli externos --update              # atualiza dólar, commodities etc.
uv run quantbt-cli corr --kind acoes --method sign --show 20
uv run quantbt-cli corr --kind acoes --preset tudo --freq Semanal --compare --out resultados/
uv run quantbt-cli corr --kind acoes --extra DOLAR BTC OURO PETROLEO
```

`uv run quantbt-cli corr --help` mostra todas as opções. Os arquivos gerados são
`ranking_pares.csv` (todos os pares, com nomes e tipos), `matriz_correlacao.csv` e `universo.csv`.

---

## 9. Atalhos

| Atalho | Ação |
|---|---|
| **Ctrl+R** | Rodar backtest e mostrar gráficos |
| **Ctrl+Shift+R** | Abrir Otimização e otimizar |
| **Ctrl+O** | Abrir arquivo CSV |
| **Ctrl+Q** | Sair |

No macOS, use **Cmd** no lugar de **Ctrl**.

---

## 10. Roteiro para começar

1. **Base B3 → Atualizar <ano atual>.**
2. **Correlação**: Tipo = *Ações / units*, Máx. = 40 → **Carregar universo** → Externos *Commodities*
   → **+ Adicionar** → Retornos *Semanal* → **Calcular correlações**.
3. Ordene por **Mais correlacionados**, clique num par interessante (ex.: PETR4 × PETROLEO) e veja no
   gráfico se a relação é estável.
4. **Abrir ativo A no backtest** → escolha *Cruzamento de médias*, Livre de risco = CDI → **Rodar**.
5. Compare com o Buy & Hold nas **Métricas**: a estratégia ganhou do CDI (Sharpe > 0)? Qual foi o
   pior drawdown?
6. **Simulação → Monte Carlo**: mesmo ativo, modelo *Histórico em blocos* → **Simular**. Veja a faixa
   de preços em 1 ano e o VaR.
7. **Simulação → Valuation**: **Buscar dados** → **Calcular preço justo**. O preço atual está acima ou
   abaixo da faixa do fluxo de caixa descontado?

---

## 11. Onde fica cada coisa

| O quê | Onde |
|---|---|
| Base da B3 | `~/.quantbt/b3/` (um arquivo por ano) |
| Externos (cache) | `~/.quantbt/externos/` |
| Suas estratégias | `~/.quantbt/strategies/` |
| Código do app | `~/projects/quantbt/quantbt/` |

Se algo der errado, a seção **Log** mostra o que aconteceu e os erros.
