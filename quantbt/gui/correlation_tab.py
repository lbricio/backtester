"""Aba de correlação entre ativos."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout,
    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from quantbt.settings import app_settings
from quantbt import b3
from quantbt import external
from quantbt import data as data_mod
from quantbt.correlation import (
    FREQS, KENDALL_MAX_ASSETS, METHOD_INFO, METHOD_LABEL, METHODS, CorrelationResult, MethodParams,
    analyze, cluster_order, rolling_measure,
)
from quantbt.gui.widgets import NumItem, Worker, _ts

SRC_B3, SRC_YAHOO = "Base B3 (COTAHIST)", "Yahoo Finance"
KINDS: dict[str, tuple[str, ...] | None] = {"Todos": None, **b3.KIND_CODES}

SHORT_LABEL = {"sign": "Direção", "pearson": "Pearson", "spearman": "Spearman", "kendall": "Kendall", "ewma": "EWMA",
               "residual": "Residual", "stress": "Estresse"}
ROLL_COLS = [("mkt_a", "A × mercado"), ("mkt_b", "B × mercado"), ("roll_mean", "Média móvel"), ("roll_std", "Desvio móvel"), ("roll_min", "Mínima móvel"),
             ("roll_last", "Atual (móvel)")]
PAIR_COLS = ["#", "Ativo A", "Ativo B", "Correlação", "Obs.", "Média móvel", "Desvio móvel",
             "Mínima móvel", "Atual (móvel)"]

PALETTE = ["#4f9dff", "#ffb454"]


def _diverging_cmap() -> pg.ColorMap:
    pos = np.array([0.0, 0.5, 1.0])
    colors = np.array([[214, 69, 65, 255], [30, 34, 41, 255], [46, 160, 255, 255]], dtype=np.ubyte)
    return pg.ColorMap(pos, colors)


def corr_color(v: float) -> QColor:
    """Cor de fundo para uma célula de correlação (vermelho negativo, azul positivo)."""
    if v is None or np.isnan(v):
        return QColor(0, 0, 0, 0)
    a = int(40 + 140 * min(abs(v), 1.0))
    return QColor(46, 160, 255, a) if v >= 0 else QColor(214, 69, 65, a)


class Heatmap(pg.PlotWidget):
    hovered = Signal(str)
    clicked = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.img = pg.ImageItem()
        self.addItem(self.img)
        self.img.setColorMap(_diverging_cmap())
        self.invertY(True)
        self.setAspectLocked(True)
        self.hideButtons()
        self.getPlotItem().setMenuEnabled(False)
        self.names: list[str] = []
        self.values: np.ndarray | None = None
        self.scene().sigMouseMoved.connect(self._on_move)
        self.scene().sigMouseClicked.connect(self._on_click)
        bar = pg.ColorBarItem(values=(-1, 1), colorMap=_diverging_cmap(), interactive=False, width=12)
        bar.setImageItem(self.img, insert_in=self.getPlotItem())

    def set_matrix(self, matrix: pd.DataFrame) -> None:
        self.names = matrix.columns.tolist()
        self.values = matrix.to_numpy(dtype=float)
        # ImageItem usa [x, y]; transposta para linha = eixo Y
        self.img.setImage(np.nan_to_num(self.values.T, nan=0.0), levels=(-1, 1))
        n = len(self.names)
        ticks = [(i + 0.5, name) for i, name in enumerate(self.names)]
        step = max(1, -(-n // 25))  # no máximo ~25 rótulos; o resto aparece ao passar o mouse
        ticks = ticks[::step]
        self.getAxis("left").setTicks([ticks, []])
        # nomes na horizontal se sobrepõem com muitos ativos: mostra só no eixo vertical
        self.getAxis("bottom").setTicks([ticks if n <= 6 else [], []])
        self.setRange(xRange=(0, n), yRange=(0, n), padding=0.02)

    def _cell(self, pos):
        if self.values is None:
            return None
        p = self.getPlotItem().vb.mapSceneToView(pos)
        i, j = int(np.floor(p.y())), int(np.floor(p.x()))
        n = len(self.names)
        if 0 <= i < n and 0 <= j < n:
            return i, j
        return None

    def _on_move(self, pos) -> None:
        c = self._cell(pos)
        if c:
            i, j = c
            self.hovered.emit(f"{self.names[i]} × {self.names[j]}: {self.values[i, j]:+.3f}")

    def _on_click(self, ev) -> None:
        c = self._cell(ev.scenePos())
        if c and c[0] != c[1]:
            self.clicked.emit(self.names[c[0]], self.names[c[1]])


class PairDetail(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.prices = self.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self.corr = self.addPlot(row=1, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self.corr.setXLink(self.prices)
        for p, t in ((self.prices, "Preços normalizados (base 100)"), (self.corr, "Correlação móvel")):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setTitle(t, size="10pt")
            p.getAxis("left").setWidth(55)
        self.prices.addLegend(offset=(10, 10))
        self.corr.setYRange(-1, 1)

    def show_pair(self, res: CorrelationResult, a: str, b: str, window: int) -> None:
        self.prices.clear()
        self.corr.clear()
        p = res.prices[[a, b]].dropna()
        if p.empty:
            return
        x = _ts(p.index)
        for name, color in zip((a, b), PALETTE):
            self.prices.plot(x, (p[name] / p[name].iloc[0] * 100).to_numpy(), pen=pg.mkPen(color, width=1.5), name=name)
        neutral = res.params.neutral if res.params else 0.0
        rc = rolling_measure(res.method, res.returns, a, b, window, neutral)
        if len(rc):
            self.corr.plot(_ts(rc.index), rc.to_numpy(), pen=pg.mkPen("#c8ccd4", width=1.5))
        full = res.matrix.at[a, b]
        self.corr.addLine(y=full, pen=pg.mkPen("#ffb454", style=Qt.DashLine))
        self.corr.addLine(y=0, pen=pg.mkPen("#555", width=1))
        if res.method == "sign":
            pct = (1 + full) / 2
            title = f"Concordância de direção móvel ({window}) — {a} × {b} — período inteiro {full:+.3f} ({pct:.0%} dos dias juntos)"
        else:
            title = f"Correlação móvel ({window}) — {a} × {b} — período inteiro {full:+.3f}"
        self.corr.setTitle(title, size="10pt")
        self.prices.autoRange()
        self.corr.setYRange(-1, 1)


class CorrelationTab(QWidget):
    log = Signal(str)
    open_in_backtest = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.settings = app_settings()
        self.pool = QThreadPool.globalInstance()
        self.result: CorrelationResult | None = None

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._controls())
        split.addWidget(self._results())
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1060])

    # ------------------------------------------------------------ layout
    def _controls(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(6)

        def grid(group: QGroupBox) -> QGridLayout:
            gl = QGridLayout(group)
            gl.setHorizontalSpacing(8)
            gl.setVerticalSpacing(5)
            gl.setColumnStretch(1, 1)
            gl.setColumnStretch(3, 1)
            return gl

        def put(gl: QGridLayout, row: int, col: int, label: str, field: QWidget, span: int = 1) -> None:
            lbl = QLabel(label)
            lbl.setBuddy(field)
            gl.addWidget(lbl, row, col * 2)
            gl.addWidget(field, row, col * 2 + 1, 1, span)
            self._labels[field] = lbl

        self._labels: dict[QWidget, QLabel] = {}

        # --- ativos
        g = QGroupBox("Ativos")
        gl = grid(g)
        self.src = QComboBox()
        self.src.addItems([SRC_B3, SRC_YAHOO])
        self.tickers = QPlainTextEdit(self.settings.value(
            "corr_tickers",
            "PETR4 PETR3 VALE3 ITUB4 BBDC4 BBAS3 ITSA4 B3SA3 ABEV3 WEGE3 ELET3 SUZB3 GGBR4 CSNA3 PRIO3 RENT3"))
        self.tickers.setPlaceholderText("Códigos separados por espaço, vírgula ou linha")
        self.tickers.setFixedHeight(64)
        put(gl, 0, 0, "Fonte", self.src, span=3)
        gl.addWidget(self.tickers, 1, 0, 1, 4)
        self.preset = QComboBox()
        self.preset.addItems(list(external.PRESETS))
        self.preset.setToolTip("Dólar, commodities, cripto e índices (Yahoo/Banco Central). "
                               "Podem ser misturados com ações da B3.")
        self.btn_preset = QPushButton("+ Adicionar")
        self.btn_preset.setToolTip("Acrescenta o grupo escolhido à lista de ativos")
        self.btn_preset.clicked.connect(self._add_preset)
        gl.addWidget(QLabel("Externos"), 2, 0)
        gl.addWidget(self.preset, 2, 1, 1, 2)
        gl.addWidget(self.btn_preset, 2, 3)
        self.in_brl = QCheckBox("Externos em R$ (converte cotações em US$ pelo dólar)")
        self.in_brl.setToolTip("Commodities e cripto cotados em dólar são convertidos para reais. "
                               "Útil para ver o efeito para o investidor brasileiro.")
        gl.addWidget(self.in_brl, 3, 0, 1, 4)
        v.addWidget(g)

        # --- universo
        g = QGroupBox("Universo da base B3")
        gl = grid(g)
        self.kind = QComboBox()
        self.kind.addItems(list(KINDS))
        self.top_n = QSpinBox()
        self.top_n.setRange(0, 100_000)
        self.top_n.setSpecialValueText("todos")
        self.top_n.setValue(0)
        self.top_n.setToolTip("Limita aos N mais líquidos (todos = sem limite)")
        self.min_vol = QDoubleSpinBox()
        self.min_vol.setRange(0, 1e9)
        self.min_vol.setDecimals(0)
        self.min_vol.setGroupSeparatorShown(True)
        self.min_vol.setSuffix(" mil")
        self.min_vol.setValue(500)
        self.min_vol.setToolTip("Volume financeiro médio mínimo por pregão no período, em milhares de R$")
        self.min_days = QSpinBox()
        self.min_days.setRange(0, 100)
        self.min_days.setSuffix(" %")
        self.min_days.setValue(90)
        self.min_days.setToolTip("Frequência mínima de negociação: % dos pregões em que o ativo teve pelo menos um negócio (desde a estreia, se 'Incluir quem estreou' estiver ligado). 90% = negociou em 9 de cada 10 pregões. Descarta ativos que passam dias sem negociar, cujo preço fica parado e depois salta.")
        put(gl, 0, 0, "Tipo", self.kind)
        put(gl, 0, 1, "Máx.", self.top_n)
        put(gl, 1, 0, "Vol. R$/dia", self.min_vol)
        put(gl, 1, 1, "Dias c/ negócio ≥", self.min_days)
        self._labels[self.min_days].setToolTip(self.min_days.toolTip())
        self.since_listing = QCheckBox("Incluir quem estreou depois do início")
        self.since_listing.setChecked(True)
        self.since_listing.setToolTip(
            "Ligado: liquidez e '% de pregões' contam a partir da estreia de cada ativo (entram IPOs e ETFs "
            "novos, como BBOV11). Desligado: exige negociação ao longo de todo o período.")
        self.min_hist = QDoubleSpinBox()
        self.min_hist.setRange(0.1, 30)
        self.min_hist.setDecimals(1)
        self.min_hist.setSingleStep(0.5)
        self.min_hist.setSuffix(" ano")
        self.min_hist.setValue(1.0)
        self.min_hist.setToolTip("Histórico mínimo desde a estreia para o ativo entrar")
        self.since_listing.toggled.connect(self.min_hist.setEnabled)
        gl.addWidget(self.since_listing, 2, 0, 1, 2)
        put(gl, 2, 1, "Hist. mín.", self.min_hist)
        self.btn_top = QPushButton("Carregar universo")
        self.btn_top.setToolTip("Preenche a lista com todos os ativos da base que passam nos filtros")
        self.btn_top.clicked.connect(self._fill_liquid)
        gl.addWidget(self.btn_top, 3, 0, 1, 4)
        v.addWidget(g)

        # --- período e método
        g = QGroupBox("Período e método")
        gl = grid(g)
        self.d_start = QDateEdit(QDate.currentDate().addYears(-10))
        self.d_end = QDateEdit(QDate.currentDate())
        for d in (self.d_start, self.d_end):
            d.setCalendarPopup(True)
            d.setDisplayFormat("dd/MM/yyyy")
        self.freq = QComboBox()
        self.freq.addItems(list(FREQS))
        self.method = QComboBox()
        self.method.addItems(list(METHODS))
        self.method.currentIndexChanged.connect(self._on_method_changed)
        self.method_info = QLabel()
        self.method_info.setWordWrap(True)
        self.method_info.setStyleSheet("color: #8b949e;")
        put(gl, 0, 0, "Início", self.d_start)
        put(gl, 0, 1, "Fim", self.d_end)
        put(gl, 1, 0, "Retornos", self.freq, span=3)
        put(gl, 2, 0, "Método", self.method, span=3)
        gl.addWidget(self.method_info, 3, 0, 1, 4)
        v.addWidget(g)

        # --- opções dos métodos (sempre visíveis; esmaecidas quando não se aplicam)
        g = QGroupBox("Opções dos métodos")
        gl = grid(g)
        self.halflife = QSpinBox()
        self.halflife.setRange(2, 5000)
        self.halflife.setValue(126)
        self.halflife.setToolTip("EWMA: meia-vida em períodos — dados dessa idade valem metade (126 pregões ≈ 6 meses)")
        self.tail = QSpinBox()
        self.tail.setRange(1, 50)
        self.tail.setValue(20)
        self.tail.setSuffix(" %")
        self.tail.setToolTip("Estresse: usa só os X% piores dias do fator de mercado")
        self.market = QLineEdit(self.settings.value("corr_market", "BOVA11"))
        self.market.setPlaceholderText("média do universo")
        self.market.setToolTip("Fator de mercado para Residual, Estresse e '× mercado' (ex.: BOVA11). "
                               "Vazio = média igualitária dos ativos analisados.")
        self.neutral = QDoubleSpinBox()
        self.neutral.setRange(0, 10)
        self.neutral.setDecimals(2)
        self.neutral.setSingleStep(0.05)
        self.neutral.setSuffix(" %")
        self.neutral.setValue(0.10)
        self.neutral.setToolTip("Direção: dias em que o ativo variou menos que isso contam como 'parado' e são ignorados")
        self.compare = QCheckBox("Comparar todos os métodos no ranking")
        self.compare.setToolTip("Calcula também os outros métodos e mostra lado a lado "
                                f"(Kendall só com até {KENDALL_MAX_ASSETS} ativos)")
        self.compare.setChecked(True)
        self.compare.toggled.connect(self._on_method_changed)
        put(gl, 0, 0, "Meia-vida", self.halflife)
        put(gl, 0, 1, "Estresse", self.tail)
        put(gl, 1, 0, "Mercado", self.market)
        put(gl, 1, 1, "Zona neutra", self.neutral)
        gl.addWidget(self.compare, 2, 0, 1, 4)
        v.addWidget(g)

        # --- avançado
        g = QGroupBox("Avançado")
        gl = grid(g)
        self.min_obs = QSpinBox()
        self.min_obs.setRange(5, 10_000)
        self.min_obs.setValue(60)
        self.min_obs.setToolTip("Mínimo de períodos em comum para um par entrar no ranking")
        self.window = QSpinBox()
        self.window.setRange(5, 2000)
        self.window.setValue(60)
        self.window.setToolTip("Janela da medida móvel (estabilidade da relação)")
        self.min_cov = QSpinBox()
        self.min_cov.setRange(0, 100)
        self.min_cov.setSuffix(" %")
        self.min_cov.setValue(50)
        self.min_cov.setToolTip("Ativos com dados em menos dessa fração do período são descartados")
        self.adjust = QCheckBox("Ajustar desdobr.")
        self.adjust.setToolTip("Corrige desdobramentos, grupamentos e bonificações de 50% detectados")
        self.adjust.setChecked(True)
        put(gl, 0, 0, "Mín. obs.", self.min_obs)
        put(gl, 0, 1, "Janela", self.window)
        put(gl, 1, 0, "Cobertura", self.min_cov)
        gl.addWidget(self.adjust, 1, 2, 1, 2)
        v.addWidget(g)
        v.addStretch(1)

        # campos se ajustam à largura do painel em vez de forçar rolagem horizontal
        for c in (self.src, self.kind, self.freq, self.method):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(w)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)

        # botão e status fixos no rodapé do painel: sempre visíveis
        self.btn_run = QPushButton("▶  Calcular correlações")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self.run)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #8b949e;")
        self.status.setMaximumHeight(80)
        self._on_method_changed()

        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(scroll, 1)
        pv.addWidget(self.btn_run)
        pv.addWidget(self.status)
        # largura mínima = o que o conteúdo precisa (nunca corta pela direita)
        need = w.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4
        panel.setMinimumWidth(max(340, need))
        return panel

    def _results(self) -> QWidget:
        vsplit = QSplitter(Qt.Vertical)
        top = QSplitter(Qt.Horizontal)

        hm_box = QWidget()
        hv = QVBoxLayout(hm_box)
        hv.setContentsMargins(0, 0, 0, 0)
        self.heatmap = Heatmap()
        self.hover_label = QLabel("Passe o mouse no mapa; clique numa célula para ver o par.")
        self.hover_label.setStyleSheet("color: #8b949e;")
        self.heatmap.hovered.connect(self.hover_label.setText)
        self.heatmap.clicked.connect(self.show_pair)
        hv.addWidget(self.heatmap, 1)
        hv.addWidget(self.hover_label)
        top.addWidget(hm_box)

        rank_box = QWidget()
        rv = QVBoxLayout(rank_box)
        rv.setContentsMargins(0, 0, 0, 0)
        h = QHBoxLayout()
        self.order = QComboBox()
        self.order.addItems(["Mais correlacionados", "Mais negativos", "Maior |correlação|", "Menos correlacionados (≈0)",
                             "Mais estáveis (menor desvio móvel)"])
        self.order.currentIndexChanged.connect(self._fill_pairs)
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filtrar por ativo…")
        self.filter.setMinimumWidth(120)
        self.order.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.order.setMinimumContentsLength(14)
        self.filter.textChanged.connect(self._fill_pairs)
        self.skip_same = QCheckBox("Ignorar mesma empresa")
        self.skip_same.setToolTip("Esconde pares do mesmo emissor (PETR3 × PETR4, mesmo prefixo de 4 letras) "
                                  "e apelidos da mesma série externa (DOLAR × WDO, IBOV × WIN)")
        self.skip_same.setChecked(True)
        self.skip_same.toggled.connect(self._fill_pairs)
        h.addWidget(self.order)
        h.addWidget(self.filter, 1)
        h.addWidget(self.skip_same)
        self.pairs_count = QLabel()
        self.pairs_count.setStyleSheet("color: #8b949e;")
        h.addWidget(self.pairs_count)
        rv.addLayout(h)
        self.pairs_table = QTableWidget(0, len(PAIR_COLS))
        self.pairs_table.setHorizontalHeaderLabels(PAIR_COLS)
        self.pairs_table.verticalHeader().setVisible(False)
        self.pairs_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.pairs_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.pairs_table.setSortingEnabled(True)
        self.pairs_table.horizontalHeader().setStretchLastSection(True)
        self.pairs_table.itemSelectionChanged.connect(self._on_pair_selected)
        rv.addWidget(self.pairs_table, 1)
        h = QHBoxLayout()
        for text, slot in (("Exportar ranking…", self._export_pairs), ("Exportar matriz…", self._export_matrix),
                           ("Abrir ativo A no backtest", self._open_a)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            h.addWidget(b)
        h.addStretch(1)
        rv.addLayout(h)
        top.addWidget(rank_box)
        top.setSizes([520, 620])

        self.detail = PairDetail()
        vsplit.addWidget(top)
        vsplit.addWidget(self.detail)
        vsplit.setSizes([520, 330])
        return vsplit

    # ------------------------------------------------------------ ações
    def _tickers(self) -> list[str]:
        raw = [t for t in re.split(r"[\s,;]+", self.tickers.toPlainText().upper()) if t]
        return list(dict.fromkeys(raw))  # remove duplicatas mantendo a ordem

    def _dates(self) -> tuple[str, str]:
        return self.d_start.date().toString("yyyy-MM-dd"), self.d_end.date().toString("yyyy-MM-dd")

    def _fill_liquid(self) -> None:
        start, end = self._dates()
        self.status.setText("Calculando liquidez…")
        self.btn_top.setEnabled(False)
        w = Worker(b3.liquid_universe, self.top_n.value(), start, end, self.min_days.value() / 100,
                   KINDS[self.kind.currentText()], self.min_vol.value() * 1000,
                   self.since_listing.isChecked(), int(self.min_hist.value() * 252))
        w.signals.finished.connect(self._on_liquid)
        w.signals.error.connect(lambda m: (self.btn_top.setEnabled(True), self._fail("Erro ao ler a base B3", m)))
        self.pool.start(w)

    def _on_liquid(self, df: pd.DataFrame) -> None:
        self.btn_top.setEnabled(True)
        if df.empty:
            self.status.setText("Nenhum ativo encontrado no período (a base cobre essas datas?).")
            return
        self.src.setCurrentText(SRC_B3)
        # mantém os externos (dólar, BTC, commodities...) que já estavam na lista
        keep_ext = [c for c in self._tickers() if external.is_external(c)]
        self.tickers.setPlainText(" ".join(list(df["ticker"]) + keep_ext))
        by_kind = ", ".join(f"{k}: {v}" for k, v in df["kind"].value_counts().items())
        kept = f" + {len(keep_ext)} externos mantidos" if keep_ext else ""
        start = pd.Timestamp(self._dates()[0])
        newer = int((pd.to_datetime(df["first"]) > start + pd.Timedelta(days=30)).sum())
        if newer:
            kept += f"; {newer} estrearam depois do início"
        self.status.setText(f"{len(df)} ativos no universo ({by_kind}){kept}. "
                            f"Volume médio de R$ {df['avg_volume'].iloc[-1] / 1e6:,.2f} a "
                            f"{df['avg_volume'].iloc[0] / 1e6:,.0f} milhões/pregão.")

    def _on_method_changed(self, *_) -> None:
        m = METHODS[self.method.currentText()]
        full = METHOD_INFO[m]
        first = full.split(". ")[0].rstrip(".") + "."
        self.method_info.setText(first + ("  ⓘ" if first != full else ""))
        self.method_info.setToolTip(full)
        self.method.setToolTip(full)
        # opções sempre visíveis; só ficam esmaecidas quando não se aplicam ao cálculo
        cmp = self.compare.isChecked()
        uses = {
            self.halflife: m == "ewma" or cmp,
            self.market: m in ("residual", "stress", "sign") or cmp,
            self.tail: m == "stress" or cmp,
            self.neutral: m == "sign" or cmp,
        }
        for wd, on in uses.items():
            wd.setEnabled(on)
            if wd in self._labels:
                self._labels[wd].setEnabled(on)

    def _add_preset(self) -> None:
        cur = self._tickers()
        add = [c for c in external.PRESETS[self.preset.currentText()] if c not in cur]
        self.tickers.setPlainText(" ".join(cur + add))
        names = ", ".join(f"{c} ({external.describe(c)})" for c in add[:4])
        self.status.setText(f"Adicionados {len(add)} ativos externos: {names}{'…' if len(add) > 4 else ''}")

    def run(self) -> None:
        tickers = self._tickers()
        if len(tickers) < 2:
            self.status.setText("Informe ao menos 2 ativos.")
            return
        self.settings.setValue("corr_tickers", " ".join(tickers))
        start, end = self._dates()
        src = self.src.currentText()
        method = METHODS[self.method.currentText()]
        compare = [m for m in METHODS.values() if m != method] if self.compare.isChecked() else []
        kwargs = dict(freq=FREQS[self.freq.currentText()], method=method, compare=compare,
                      min_obs=self.min_obs.value(), rolling_window=self.window.value(),
                      min_coverage=self.min_cov.value() / 100)
        adjust = self.adjust.isChecked()
        mkt = self.market.text().strip().upper()
        self.settings.setValue("corr_market", mkt)
        mparams = MethodParams(halflife=self.halflife.value(), tail_q=self.tail.value() / 100, market_name=mkt,
                               neutral=self.neutral.value() / 100)

        in_brl = self.in_brl.isChecked()

        def job():
            b3_loader = (lambda ts: b3.load_closes(ts, start, end, adjust=adjust)) if src == SRC_B3 else (
                lambda ts: data_mod.load_yahoo_closes(ts, start, end))

            def load(ts):
                return external.combined_closes(ts, b3_loader, start, end, in_brl=in_brl)
            prices = load(tickers)
            missing = [t for t in tickers if t not in prices.columns]
            notes = []
            if mkt:
                try:
                    mparams.market = prices[mkt] if mkt in prices else load([mkt])[mkt]
                except Exception:
                    notes.append(f"Fator de mercado {mkt} não encontrado: usando a média do universo.")
            ext_in = [c for c in prices.columns if external.is_external(c)]
            if ext_in and kwargs["freq"] is None:
                notes.append("Externos fecham em horários diferentes da B3 (commodities e S&P nos EUA, PTAX às "
                             "13h): no diário a correlação fica subestimada. Compare com retornos semanais.")
            res = analyze(prices, params=mparams, **kwargs)
            res.notes = notes + res.notes
            return res, missing

        self.btn_run.setEnabled(False)
        self.status.setText(f"Carregando {len(tickers)} ativos e calculando…")
        w = Worker(job)
        w.signals.finished.connect(self._on_done)
        w.signals.error.connect(lambda m: self._fail("Erro no cálculo de correlação", m))
        self.pool.start(w)

    def _fail(self, title: str, msg: str) -> None:
        self.btn_run.setEnabled(True)
        short = msg.split("\n\nTraceback")[0]
        self.status.setText(f"{title}: {short}")
        self.log.emit(f"ERRO — {title}: {short}")

    def _on_done(self, out) -> None:
        res, missing = out
        self.btn_run.setEnabled(True)
        self.result = res
        order = cluster_order(res.matrix)
        self.heatmap.set_matrix(res.matrix.loc[order, order])
        self._fill_pairs()
        n = res.matrix.shape[0]
        msg = (f"{METHOD_LABEL[res.method]}: {n} ativos, {len(res.pairs):,} pares, {len(res.returns):,} períodos "
               f"({res.returns.index[0]:%d/%m/%Y} → {res.returns.index[-1]:%d/%m/%Y}).")
        def short_list(items, n=8):
            return ", ".join(items[:n]) + (f" e mais {len(items) - n}" if len(items) > n else "")
        if missing:
            msg += f"\nNão encontrados ({len(missing)}): {short_list(missing)}."
        if res.dropped:
            msg += f"\nDescartados por pouca cobertura ({len(res.dropped)}): {short_list(res.dropped)}."
        for note in res.notes:
            msg += f"\n{note}"
        self.status.setText(msg)
        self.log.emit("Correlação: " + msg.replace("\n", " "))
        shown = self._sorted_pairs()
        if len(shown):
            top = shown.iloc[0]
            self.show_pair(top["a"], top["b"])

    def _sorted_pairs(self) -> pd.DataFrame:
        p = self.result.pairs
        if self.skip_same.isChecked() and "same_issuer" in p:
            p = p[~p["same_issuer"]]
        f = self.filter.text().strip().upper()
        if f:
            p = p[p["a"].str.contains(f, regex=False) | p["b"].str.contains(f, regex=False)]
        mode = self.order.currentIndex()
        if mode == 1:
            p = p.sort_values("corr")
        elif mode == 2:
            p = p.reindex(p["corr"].abs().sort_values(ascending=False).index)
        elif mode == 3:
            p = p.reindex(p["corr"].abs().sort_values().index)
        elif mode == 4 and "roll_std" in p:
            p = p.sort_values("roll_std")
        return p

    def _fill_pairs(self) -> None:
        if self.result is None:
            return
        full = self._sorted_pairs()
        p = full.head(2000)
        self.pairs_count.setText(f"{len(p):,} de {len(full):,} pares")
        res = self.result
        cmp_cols = [f"corr_{m}" for m in res.compared if f"corr_{m}" in p]
        roll = [(k, lbl) for k, lbl in ROLL_COLS if k in p]
        headers = (["#", "Ativo A", "Ativo B", SHORT_LABEL[res.method], "Obs."]
                   + [SHORT_LABEL[c[5:]] for c in cmp_cols] + [lbl for _, lbl in roll])
        t = self.pairs_table
        t.setSortingEnabled(False)
        t.clear()
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(p))
        a_s, b_s, corr_s, obs_s = p["a"].tolist(), p["b"].tolist(), p["corr"].to_numpy(), p["obs"].to_numpy()
        extra = [p[c].to_numpy() for c in cmp_cols]
        rolls = [(k, p[k].to_numpy()) for k, _ in roll]
        for r in range(len(p)):
            corr = corr_s[r]
            cells = [
                NumItem(str(r + 1), r + 1),
                self._name_item(a_s[r]),
                self._name_item(b_s[r]),
                NumItem(f"{corr:+.3f}", corr),
                NumItem(f"{obs_s[r]:,}", obs_s[r]),
            ]
            cells[3].setBackground(corr_color(corr))
            for arr in extra:
                val = arr[r]
                item = NumItem("—" if pd.isna(val) else f"{val:+.3f}", val)
                item.setBackground(corr_color(val))
                cells.append(item)
            for k, arr in rolls:
                val = arr[r]
                if pd.isna(val):
                    text = "—"
                elif k.startswith("mkt_"):
                    text = f"{val:.0%}"
                else:
                    text = f"{val:+.3f}" if k != "roll_std" else f"{val:.3f}"
                cells.append(NumItem(text, val))
            for c, item in enumerate(cells):
                t.setItem(r, c, item)
        t.resizeColumnsToContents()
        t.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        t.setSortingEnabled(True)

    @staticmethod
    def _name_item(code: str) -> QTableWidgetItem:
        item = QTableWidgetItem(code)
        name = external.describe(code)
        if name:
            item.setToolTip(name)
            item.setForeground(QColor("#ffb454"))  # destaca ativos externos
        return item

    def _selected_pair(self) -> tuple[str, str] | None:
        rows = self.pairs_table.selectionModel().selectedRows()
        if not rows:
            return None
        r = rows[0].row()
        return self.pairs_table.item(r, 1).text(), self.pairs_table.item(r, 2).text()

    def _on_pair_selected(self) -> None:
        pair = self._selected_pair()
        if pair:
            self.show_pair(*pair)

    def show_pair(self, a: str, b: str) -> None:
        if self.result is not None:
            self.detail.show_pair(self.result, a, b, self.result.window or self.window.value())

    def _open_a(self) -> None:
        pair = self._selected_pair()
        if pair:
            self.open_in_backtest.emit(pair[0])

    def _export_pairs(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar ranking", "ranking_correlacao.csv", "CSV (*.csv)")
        if path:
            self._sorted_pairs().to_csv(path, index=False)  # exporta a visão atual (ordem e filtros)
            self.log.emit(f"Ranking exportado para {path}")

    def _export_matrix(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar matriz", "matriz_correlacao.csv", "CSV (*.csv)")
        if path:
            self.result.matrix.to_csv(path)
            self.log.emit(f"Matriz exportada para {path}")
