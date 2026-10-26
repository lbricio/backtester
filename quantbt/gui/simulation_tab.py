"""Seção Simulação: Monte Carlo do preço, preço justo de opções e valuation de ações."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, QSettings, Qt, QThreadPool, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QCompleter, QDateEdit, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from quantbt.settings import app_settings
from quantbt import external, sources
from quantbt import montecarlo as mc
from quantbt import options as op
from quantbt import valuation as va
from quantbt.gui.widgets import C_BUY, C_SELL, NumItem, Worker, _ts

C_PATH = (79, 157, 255, 38)
C_MEDIAN = "#ffb454"
C_HIST = "#d7dae0"
C_TARGET = "#2ecc71"
C_NOW = "#ff7b72"
MAX_DRAWN = 150  # trajetórias desenhadas (o resto entra só nas estatísticas)
MAX_STATS_DURING_ANIM = 5_000  # amostra usada nos percentis durante a animação


# ---------------------------------------------------------------- utilidades de layout
def _dspin(lo, hi, val, dec=2, suffix="", step=None, tip=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(dec)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    if step:
        w.setSingleStep(step)
    if tip:
        w.setToolTip(tip)
    w.setGroupSeparatorShown(True)
    return w


def _ispin(lo, hi, val, suffix="", step=1, tip=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    w.setValue(val)
    w.setSingleStep(step)
    if suffix:
        w.setSuffix(suffix)
    if tip:
        w.setToolTip(tip)
    w.setGroupSeparatorShown(True)
    return w


class _Panel:
    """Painel de controles com grupos em grade (2 campos por linha) e botões fixos no rodapé."""

    def __init__(self) -> None:
        self.body = QWidget()
        self.v = QVBoxLayout(self.body)
        self.v.setSpacing(6)
        self.labels: dict[QWidget, QLabel] = {}

    def group(self, title: str) -> QGridLayout:
        g = QGroupBox(title)
        gl = QGridLayout(g)
        gl.setHorizontalSpacing(8)
        gl.setVerticalSpacing(5)
        gl.setColumnStretch(1, 1)
        gl.setColumnStretch(3, 1)
        self.v.addWidget(g)
        return gl

    def put(self, gl: QGridLayout, row: int, col: int, label: str, field: QWidget, span: int = 1) -> None:
        lbl = QLabel(label)
        gl.addWidget(lbl, row, col * 2)
        gl.addWidget(field, row, col * 2 + 1, 1, span)
        self.labels[field] = lbl

    def finish(self, footer: list[QWidget], min_width: int = 360) -> QWidget:
        self.v.addStretch(1)
        for c in self.body.findChildren(QComboBox):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(6)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.body)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(scroll, 1)
        for w in footer:
            pv.addWidget(w)
        need = self.body.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4
        panel.setMinimumWidth(max(min_width, need))
        return panel


def _run_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setMinimumHeight(36)
    b.setStyleSheet("font-weight: bold;")
    return b


def _status() -> QLabel:
    s = QLabel()
    s.setWordWrap(True)
    s.setStyleSheet("color: #8b949e;")
    s.setMaximumHeight(90)
    return s


def _ticker_field(settings: QSettings, key: str, default: str) -> QLineEdit:
    e = QLineEdit(settings.value(key, default))
    e.setToolTip("Código da B3 (PETR4) ou ativo externo (DOLAR, BTC, OURO, IBOV...)")
    comp = QCompleter(list(external.CATALOG), e)
    comp.setCaseSensitivity(Qt.CaseInsensitive)
    e.setCompleter(comp)
    return e


def _stats_table() -> QTableWidget:
    t = QTableWidget(0, 2)
    t.setHorizontalHeaderLabels(["Medida", "Valor"])
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _fill_rows(t: QTableWidget, rows: list[tuple[str, str]], colors: dict[int, str] | None = None) -> None:
    t.setRowCount(len(rows))
    for r, (k, v) in enumerate(rows):
        t.setItem(r, 0, QTableWidgetItem(k))
        item = QTableWidgetItem(v)
        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        if colors and r in colors:
            item.setForeground(QColor(colors[r]))
        t.setItem(r, 1, item)


def _money(v: float) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:,.2f}"


def _pct(v: float | None, signed: bool = True) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:+.1%}" if signed else f"{v:.1%}"


def _paths_xy(x: np.ndarray, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Várias trajetórias num único item do gráfico (separadas por NaN): bem mais rápido."""
    k = len(paths)
    xs = np.tile(np.r_[x, np.nan], k)
    ys = np.hstack([paths, np.full((k, 1), np.nan)]).ravel()
    return xs, ys


class _Animator:
    """Executa passos com um QTimer e controla pausa/parada."""

    def __init__(self, on_step, on_end) -> None:
        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.on_step, self.on_end = on_step, on_end
        self.running = False

    def start(self, interval_ms: int) -> None:
        self.running = True
        self.timer.start(interval_ms)

    def pause(self) -> bool:
        if self.timer.isActive():
            self.timer.stop()
            return True
        if self.running:
            self.timer.start()
        return False

    def stop(self) -> None:
        self.timer.stop()
        if self.running:
            self.running = False
            self.on_end()

    def _tick(self) -> None:
        if not self.on_step():
            self.stop()


def _speed_slider() -> QSlider:
    s = QSlider(Qt.Horizontal)
    s.setRange(1, 10)
    s.setValue(6)
    s.setToolTip("Velocidade da animação (10 = quase instantâneo)")
    return s


def _interval(slider: QSlider) -> int:
    return int(np.interp(slider.value(), [1, 10], [350, 1]))


# ================================================================ Monte Carlo do preço
class MonteCarloTab(QWidget):
    def __init__(self, log, get_rf, parent=None) -> None:
        super().__init__(parent)
        self.log, self.get_rf = log, get_rf
        self.settings = app_settings()
        self.pool = QThreadPool.globalInstance()
        self.est: mc.Estimate | None = None
        self.hist: pd.Series | None = None
        self.paths: np.ndarray | None = None
        self.count = 0
        self.anim = _Animator(self._step, self._finish)

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._controls())
        split.addWidget(self._results())
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1060])

    # ------------------------------------------------------------ layout
    def _controls(self) -> QWidget:
        p = _Panel()
        gl = p.group("Ativo")
        self.src = QComboBox()
        self.src.addItems([sources.SRC_B3, sources.SRC_YAHOO])
        self.ticker = _ticker_field(self.settings, "mc_ticker", "PETR4")
        self.years = _dspin(0.5, 20, 2, 1, " anos", 0.5, "Período usado para estimar volatilidade e tendência")
        p.put(gl, 0, 0, "Fonte", self.src, span=3)
        p.put(gl, 1, 0, "Ativo", self.ticker)
        p.put(gl, 1, 1, "Histórico", self.years)

        gl = p.group("Modelo")
        self.model = QComboBox()
        self.model.addItems(list(mc.MODELS))
        self.model.currentIndexChanged.connect(self._on_model)
        self.model_info = QLabel()
        self.model_info.setWordWrap(True)
        self.model_info.setStyleSheet("color: #8b949e;")
        self.drift = QComboBox()
        self.drift.addItems(list(mc.DRIFTS))
        self.drift.setToolTip("Histórica: repete a tendência do período. Cuidado: com 2 anos e 30% de volatilidade, "
                              "a margem de erro dessa média é de ±21 pontos por ano.\n"
                              "Taxa livre de risco: rende em média o CDI.\nSem tendência: preço esperado = atual.\n"
                              "Personalizada: o retorno anual que você informar.")
        # padrão conservador: a média histórica de poucos anos é muito imprecisa (±σ/√anos por ano)
        self.drift.setCurrentText("Taxa livre de risco")
        self.drift.currentIndexChanged.connect(self._on_model)
        self.custom = _dspin(-90, 500, 10, 1, " % a.a.", tip="Retorno anual esperado (tendência personalizada)")
        self.vol = _dspin(0, 500, 0, 1, " % a.a.", tip="0 = volatilidade histórica. Aumente para simular estresse.")
        self.vol.setSpecialValueText("histórica")
        self.rf = _dspin(0, 100, 0, 2, " % a.a.", tip="Taxa livre de risco (tendência 'Taxa livre de risco')")
        self.block = _ispin(2, 250, 20, " dias", tip="Tamanho dos blocos (modelo Histórico em blocos)")
        p.put(gl, 0, 0, "Modelo", self.model, span=3)
        gl.addWidget(self.model_info, 1, 0, 1, 4)
        p.put(gl, 2, 0, "Tendência", self.drift, span=3)
        p.put(gl, 3, 0, "Retorno", self.custom)
        p.put(gl, 3, 1, "Volatilidade", self.vol)
        p.put(gl, 4, 0, "Livre risco", self.rf)
        p.put(gl, 4, 1, "Bloco", self.block)

        gl = p.group("Simulação")
        self.horizon = _ispin(5, 2520, 252, " pregões", 21, "252 pregões ≈ 1 ano")
        self.n_paths = _ispin(100, mc.MAX_PATHS, 5000, step=1000)
        self.target = _dspin(0, 1e9, 0, 2, tip="Preço-alvo opcional: o app calcula a chance de chegar nele")
        self.target.setSpecialValueText("nenhum")
        self.seed = _ispin(0, 10**9, 0, tip="0 = aleatória a cada rodada; outro número repete os mesmos sorteios")
        self.seed.setSpecialValueText("aleatória")
        p.put(gl, 0, 0, "Horizonte", self.horizon)
        p.put(gl, 0, 1, "Simulações", self.n_paths)
        p.put(gl, 1, 0, "Preço-alvo", self.target)
        p.put(gl, 1, 1, "Semente", self.seed)

        gl = p.group("Animação")
        self.speed = _speed_slider()
        p.put(gl, 0, 0, "Velocidade", self.speed, span=3)

        self.btn_run = _run_button("▶  Simular")
        self.btn_run.clicked.connect(self.run)
        h = QHBoxLayout()
        self.btn_pause = QPushButton("⏸ Pausar")
        self.btn_pause.clicked.connect(self._pause)
        self.btn_stop = QPushButton("■ Parar")
        self.btn_stop.clicked.connect(self.anim.stop)
        self.btn_export = QPushButton("Exportar percentis…")
        self.btn_export.clicked.connect(self._export)
        for b in (self.btn_pause, self.btn_stop, self.btn_export):
            h.addWidget(b)
        hw = QWidget()
        hw.setLayout(h)
        h.setContentsMargins(0, 0, 0, 0)
        self.status = _status()
        self._on_model()
        self._set_running(False)
        return p.finish([self.btn_run, hw, self.status])

    def _results(self) -> QWidget:
        vs = QSplitter(Qt.Vertical)
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self.hplot = self.glw.addPlot(row=0, col=1)
        self.glw.ci.layout.setColumnStretchFactor(0, 5)
        self.glw.ci.layout.setColumnStretchFactor(1, 1)
        self.plot.showGrid(x=True, y=True, alpha=0.15)
        self.plot.setTitle("Trajetórias simuladas", size="10pt")
        self.plot.getAxis("left").setWidth(70)
        self.plot.addLegend(offset=(10, 10))
        self.hplot.setYLink(self.plot)
        self.hplot.setTitle("Preço no fim", size="10pt")
        self.hplot.hideAxis("left")
        self.hplot.showGrid(x=True, alpha=0.1)
        vs.addWidget(self.glw)

        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        self.table = _stats_table()
        bl.addWidget(self.table, 1)
        self.explain = QLabel(
            "<b>Como ler:</b> cada linha azul é um futuro possível. A faixa clara contém 90% das trajetórias "
            "(entre os percentis 5 e 95) e a escura 50% (25 a 75). A linha laranja é a mediana. À direita, a "
            "distribuição do preço no fim do horizonte.<br><br><b>Cuidado:</b> a simulação supõe que o futuro "
            "terá a mesma volatilidade (e, na tendência histórica, o mesmo rumo) do período escolhido. Ela mostra "
            "a faixa de incerteza, não uma previsão.")
        self.explain.setWordWrap(True)
        self.explain.setStyleSheet("color: #8b949e;")
        self.explain.setAlignment(Qt.AlignTop)
        bl.addWidget(self.explain, 1)
        vs.addWidget(bottom)
        vs.setSizes([560, 280])
        self._init_items()
        return vs

    def _init_items(self) -> None:
        self.plot.clear()
        self.plot.legend.clear()
        self.hplot.clear()
        self.hist_item = self.plot.plot(pen=pg.mkPen(C_HIST, width=1.4), name="Histórico")
        self.paths_item = pg.PlotDataItem(pen=pg.mkPen(C_PATH, width=1), connect="finite")
        self.paths_item.setZValue(0)
        self.plot.addItem(self.paths_item)
        # contornos das faixas: curvas finas (o preenchimento precisa de curvas com caminho desenhável)
        edge = {5: (255, 180, 84, 120), 95: (255, 180, 84, 120), 25: (255, 180, 84, 60), 75: (255, 180, 84, 60)}
        self.c = {p: pg.PlotCurveItem(pen=pg.mkPen(edge.get(p, (0, 0, 0, 0)), width=1)) for p in mc.PCTS}
        for item in self.c.values():
            item.setZValue(6)
            self.plot.addItem(item)
        self.band90 = pg.FillBetweenItem(self.c[5], self.c[95], brush=pg.mkBrush(255, 180, 84, 40))
        self.band50 = pg.FillBetweenItem(self.c[25], self.c[75], brush=pg.mkBrush(255, 180, 84, 55))
        for z, band in ((4, self.band90), (5, self.band50)):
            band.setZValue(z)  # faixas por cima das trajetórias, semitransparentes
            self.plot.addItem(band)
        self.median_item = self.plot.plot(pen=pg.mkPen(C_MEDIAN, width=2.5), name="Mediana")
        self.median_item.setZValue(10)
        legend = self.plot.legend
        legend.addItem(pg.BarGraphItem(x=[0], height=[0], width=0, brush=pg.mkBrush(255, 180, 84, 95)), "50% das trajetórias")
        legend.addItem(pg.BarGraphItem(x=[0], height=[0], width=0, brush=pg.mkBrush(255, 180, 84, 40)), "90% das trajetórias")
        self.target_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(C_TARGET, width=2, style=Qt.DashLine),
                                           label="alvo", labelOpts={"color": C_TARGET, "position": 0.05})
        self.target_line.setZValue(20)
        self.now_line = pg.InfiniteLine(angle=0, pen=pg.mkPen(C_NOW, width=1, style=Qt.DotLine))
        self.now_line.setZValue(20)
        self.plot.addItem(self.now_line)
        self.bars = pg.BarGraphItem(x0=0, y=[0], width=[0], height=1, brush=pg.mkBrush(79, 157, 255, 160))
        self.hplot.addItem(self.bars)
        self.hist_now = pg.InfiniteLine(angle=0, pen=pg.mkPen(C_NOW, width=1, style=Qt.DotLine))
        self.hplot.addItem(self.hist_now)

    # ------------------------------------------------------------ ações
    def _on_model(self, *_) -> None:
        m = mc.MODELS[self.model.currentText()]
        self.model_info.setText(mc.MODEL_INFO[m])
        d = mc.DRIFTS[self.drift.currentText()]
        self.custom.setEnabled(d == "custom")
        self.rf.setEnabled(d == "risk_free")
        self.block.setEnabled(m == "block")

    def _set_running(self, on: bool) -> None:
        self.btn_run.setEnabled(not on)
        self.btn_pause.setEnabled(on)
        self.btn_stop.setEnabled(on)
        self.btn_pause.setText("⏸ Pausar")

    def _pause(self) -> None:
        paused = self.anim.pause()
        self.btn_pause.setText("▶ Continuar" if paused else "⏸ Pausar")

    def config(self) -> mc.MCConfig:
        return mc.MCConfig(
            model=mc.MODELS[self.model.currentText()], horizon=self.horizon.value(), n_paths=self.n_paths.value(),
            drift=mc.DRIFTS[self.drift.currentText()], custom_return=self.custom.value() / 100,
            risk_free=self.rf.value() / 100, vol_override=self.vol.value() / 100, block=self.block.value(),
            seed=self.seed.value() or None,
        )

    def run(self) -> None:
        t = self.ticker.text().strip()
        if not t:
            return
        if self.rf.value() == 0 and self.get_rf() > 0:
            self.rf.setValue(self.get_rf() * 100)
        self.settings.setValue("mc_ticker", t)
        years, src = self.years.value(), self.src.currentText()
        start = str((pd.Timestamp.today() - pd.DateOffset(days=int(365.25 * (years + 0.5)))).date())
        self._set_running(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)  # só há o que parar depois que a simulação começa
        self.status.setText(f"Carregando {t}…")
        w = Worker(sources.load_prices, t, src, start)
        w.signals.finished.connect(lambda df: self._start(df, t, years))
        w.signals.error.connect(self._fail)
        self.pool.start(w)

    def _fail(self, msg: str) -> None:
        self._set_running(False)
        short = msg.split("\n\nTraceback")[0]
        self.status.setText(f"Erro: {short}")
        self.log(f"ERRO — Monte Carlo: {short}")

    def _start(self, df: pd.DataFrame, ticker: str, years: float) -> None:
        try:
            self.est = mc.estimate(df["close"], years)
        except ValueError as e:
            self._fail(str(e))
            return
        cfg = self.cfg = self.config()
        est = self.est
        self.ticker_name = ticker.upper()
        self.hist = df["close"][df.index >= est.last_date - pd.DateOffset(days=int(365.25 * years))]
        self.dates = mc.future_dates(est.last_date, cfg.horizon, est.ppy)
        self.x = _ts(self.dates)
        total = min(cfg.n_paths, mc.MAX_PATHS)
        self.paths = np.empty((total, cfg.horizon + 1), dtype=np.float32)
        self.count = 0
        n_batches = int(np.interp(self.speed.value(), [1, 10], [80, 8]))
        self.gen = mc.simulate_batches(est, cfg, batches=n_batches)
        self.drawn = np.empty((0, cfg.horizon + 1), dtype=np.float32)
        mu_d, sig_d = mc.daily_params(est, cfg)
        self.used = (float(np.exp((mu_d + sig_d**2 / 2) * est.ppy) - 1), sig_d * np.sqrt(est.ppy))

        self._init_items()
        self.hist_item.setData(_ts(self.hist.index), self.hist.to_numpy())
        self.now_line.setValue(est.s0)
        self.hist_now.setValue(est.s0)
        if self.target.value() > 0:
            self.target_line.setValue(self.target.value())
            self.plot.addItem(self.target_line)
        self.plot.setTitle(f"{sources.describe(self.ticker_name)} — {total:,} trajetórias, "
                           f"{cfg.horizon} pregões à frente ({self.model.currentText()})", size="10pt")
        lo, hi = float(self.hist.min()), float(self.hist.max())
        self.plot.setXRange(_ts(self.hist.index[:1])[0], self.x[-1], padding=0.02)
        self.plot.setYRange(lo * 0.8, hi * 1.25, padding=0)
        self.plot.enableAutoRange(axis="y", enable=False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.log(f"Monte Carlo {self.ticker_name}: {total:,} trajetórias, {cfg.horizon} pregões, "
                 f"σ {self.used[1]:.1%} a.a., retorno esperado {self.used[0]:+.1%} a.a.")
        self.anim.start(_interval(self.speed))

    def _step(self) -> bool:
        try:
            batch = next(self.gen)
        except StopIteration:
            return False
        n = len(batch)
        self.paths[self.count:self.count + n] = batch
        self.count += n
        total = len(self.paths)
        room = MAX_DRAWN - len(self.drawn)
        if room > 0:  # novas trajetórias aparecem no gráfico
            take = max(1, int(np.ceil(MAX_DRAWN * n / total)))
            self.drawn = np.vstack([self.drawn, batch[:min(room, take)]])
            self.paths_item.setData(*_paths_xy(self.x, self.drawn))
        sample = self.paths[:self.count]
        if self.count > MAX_STATS_DURING_ANIM:
            sample = sample[np.linspace(0, self.count - 1, MAX_STATS_DURING_ANIM).astype(int)]
        self._update(sample, final=False)
        return True

    def _finish(self) -> None:
        self._set_running(False)
        if self.paths is None or self.count == 0:
            return
        self.paths = self.paths[:self.count]
        st = self._update(self.paths, final=True)
        self.last_stats = st
        self.status.setText(f"Pronto: {self.count:,} trajetórias. Mediana no fim: {_money(st.median_final)} "
                            f"({st.median_final / st.s0 - 1:+.1%}); 90% entre {_money(st.pcts[5][-1])} e "
                            f"{_money(st.pcts[95][-1])}.")

    def _update(self, sample: np.ndarray, final: bool) -> mc.MCStats:
        st = mc.stats(sample, self.est.s0, self.target.value() or None)
        for p in mc.PCTS:
            self.c[p].setData(self.x, st.pcts[p])
        self.median_item.setData(self.x, st.pcts[50])
        # histograma do preço final (barras horizontais, alinhadas ao eixo de preço)
        lo, hi = np.percentile(st.final, [0.5, 99.5])
        counts, edges = np.histogram(st.final, bins=45, range=(lo, hi))
        centers = (edges[:-1] + edges[1:]) / 2
        self.bars.setOpts(x0=0, y=centers, width=counts / max(counts.sum(), 1) * 100,
                          height=(edges[1] - edges[0]) * 0.9)
        self.hplot.getAxis("bottom").setLabel("% das simulações")
        top = max(float(np.max(self.drawn)) if len(self.drawn) else 0, float(st.pcts[95].max()))
        bottom = min(float(np.min(self.drawn)) if len(self.drawn) else top, float(st.pcts[5].min()))
        lo_y = min(bottom, float(self.hist.min())) * 0.95
        hi_y = max(top, float(self.hist.max()), self.target.value()) * 1.05
        self.plot.setYRange(lo_y, hi_y, padding=0)
        self._fill_table(st, final)
        if not final:
            self.status.setText(f"Simulando… {self.count:,} de {len(self.paths):,} trajetórias")
        return st

    def _fill_table(self, st: mc.MCStats, final: bool) -> None:
        s0 = st.s0
        rows = [
            ("Simulações", f"{self.count:,} de {len(self.paths):,}" + ("" if final else " (parcial)")),
            ("Preço atual", f"{_money(s0)} em {self.est.last_date:%d/%m/%Y}"),
            ("Fim do horizonte", f"{self.dates[-1]:%d/%m/%Y} ({self.cfg.horizon} pregões)"),
            ("Mediana no fim", f"{_money(st.median_final)} ({st.median_final / s0 - 1:+.1%})"),
            ("Média no fim", f"{_money(st.mean_final)} ({st.mean_final / s0 - 1:+.1%})"),
            ("Faixa de 90% (P5 – P95)", f"{_money(st.pcts[5][-1])} – {_money(st.pcts[95][-1])}"),
            ("Faixa de 50% (P25 – P75)", f"{_money(st.pcts[25][-1])} – {_money(st.pcts[75][-1])}"),
            ("Chance de terminar acima do atual", _pct(st.prob_up, False)),
            ("Perda no pior 5% (VaR 95%)", _pct(st.var95)),
            ("Perda média no pior 5% (CVaR 95%)", _pct(st.cvar95)),
            ("Queda máxima típica no caminho", _pct(st.median_max_dd)),
            ("Volatilidade usada", f"{self.used[1]:.1%} a.a. (histórica {self.est.sigma_annual:.1%})"),
            ("Retorno esperado usado", f"{self.used[0]:+.1%} a.a. (histórico {self.est.mu_annual:+.1%})"),
        ]
        colors = {7: C_BUY if st.prob_up >= 0.5 else C_SELL, 8: C_SELL, 9: C_SELL}
        if st.target:
            side = "acima" if st.target >= s0 else "abaixo"
            rows += [(f"Chance de terminar {side} do alvo {_money(st.target)}", _pct(st.prob_above_target, False)),
                     ("Chance de tocar o alvo em algum momento", _pct(st.prob_touch_target, False))]
        _fill_rows(self.table, rows, colors)

    def _export(self) -> None:
        if self.paths is None or self.count == 0:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar percentis", "monte_carlo_percentis.csv", "CSV (*.csv)")
        if path:
            st = mc.stats(self.paths[:self.count], self.est.s0)
            pd.DataFrame({f"p{p}": st.pcts[p] for p in mc.PCTS}, index=self.dates).to_csv(path)
            self.log(f"Percentis do Monte Carlo exportados para {path}")


# ================================================================ opções
class OptionsTab(QWidget):
    def __init__(self, log, get_rf, parent=None) -> None:
        super().__init__(parent)
        self.log, self.get_rf = log, get_rf
        self.settings = app_settings()
        self.pool = QThreadPool.globalInstance()
        self.pricer: op.MCOptionPricer | None = None
        self.anim = _Animator(self._step, self._finish)

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._controls())
        split.addWidget(self._results())
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1060])

    def _controls(self) -> QWidget:
        p = _Panel()
        gl = p.group("Ativo-objeto")
        self.src = QComboBox()
        self.src.addItems([sources.SRC_B3, sources.SRC_YAHOO])
        self.ticker = _ticker_field(self.settings, "opt_ticker", "PETR4")
        self.years = _dspin(0.25, 10, 1, 2, " anos", 0.25, "Período para a volatilidade histórica")
        self.btn_fill = QPushButton("Usar preço e volatilidade do ativo")
        self.btn_fill.clicked.connect(self._fill_from_asset)
        p.put(gl, 0, 0, "Fonte", self.src, span=3)
        p.put(gl, 1, 0, "Ativo", self.ticker)
        p.put(gl, 1, 1, "Histórico", self.years)
        gl.addWidget(self.btn_fill, 2, 0, 1, 4)

        gl = p.group("Opção")
        self.kind = QComboBox()
        self.kind.addItems(["Compra (call)", "Venda (put)"])
        self.s = _dspin(0.01, 1e9, 30, 2, tip="Preço atual do ativo-objeto")
        self.k = _dspin(0.01, 1e9, 30, 2, tip="Preço de exercício")
        self.expiry = QDateEdit(QDate.currentDate().addMonths(2))
        self.expiry.setCalendarPopup(True)
        self.expiry.setDisplayFormat("dd/MM/yyyy")
        self.days = _ispin(0, 2520, 42, " d.u.", tip="Dias úteis até o vencimento (feriados não descontados)")
        self.expiry.dateChanged.connect(self._on_expiry)
        self._on_expiry()
        p.put(gl, 0, 0, "Tipo", self.kind, span=3)
        p.put(gl, 1, 0, "Ativo (S)", self.s)
        p.put(gl, 1, 1, "Strike (K)", self.k)
        p.put(gl, 2, 0, "Vencimento", self.expiry)
        p.put(gl, 2, 1, "Prazo", self.days)

        gl = p.group("Mercado")
        self.r = _dspin(0, 100, 0, 2, " % a.a.", tip="Taxa livre de risco (CDI). Use o botão para buscar o atual.")
        self.btn_cdi = QPushButton("CDI atual")
        self.btn_cdi.setToolTip("Busca o CDI anualizado no Banco Central")
        self.btn_cdi.clicked.connect(self._fetch_cdi)
        self.sigma = _dspin(0.1, 500, 30, 1, " % a.a.", tip="Volatilidade anual usada no preço")
        self.q = _dspin(0, 100, 0, 2, " % a.a.", tip="Dividend yield esperado até o vencimento")
        self.market = _dspin(0, 1e9, 0, 2, tip="Preço da opção no mercado (opcional): calcula a vol. implícita")
        self.market.setSpecialValueText("—")
        p.put(gl, 0, 0, "Juros (r)", self.r)
        gl.addWidget(self.btn_cdi, 0, 2, 1, 2)
        p.put(gl, 1, 0, "Volatilidade", self.sigma)
        p.put(gl, 1, 1, "Dividendos", self.q)
        p.put(gl, 2, 0, "Preço mercado", self.market)

        gl = p.group("Monte Carlo")
        self.n = _ispin(10_000, 5_000_000, 400_000, step=100_000)
        self.speed = _speed_slider()
        p.put(gl, 0, 0, "Simulações", self.n, span=3)
        p.put(gl, 1, 0, "Velocidade", self.speed, span=3)

        self.btn_run = _run_button("▶  Calcular preço justo")
        self.btn_run.clicked.connect(self.run)
        self.status = _status()
        return p.finish([self.btn_run, self.status])

    def _results(self) -> QWidget:
        vs = QSplitter(Qt.Vertical)
        glw = pg.GraphicsLayoutWidget()
        self.conv = glw.addPlot(row=0, col=0)
        self.dist = glw.addPlot(row=0, col=1)
        self.conv.setTitle("Convergência: preço por Monte Carlo × Black-Scholes", size="10pt")
        self.conv.setLabel("bottom", "simulações")
        self.conv.setLogMode(x=True, y=False)
        self.conv.showGrid(x=True, y=True, alpha=0.15)
        self.conv.addLegend(offset=(10, 10))
        self.dist.setTitle("Preço do ativo no vencimento (simulado)", size="10pt")
        self.dist.showGrid(x=True, y=True, alpha=0.1)
        self.dist.setLabel("left", "% das simulações")
        vs.addWidget(glw)
        bottom = QWidget()
        bl = QHBoxLayout(bottom)
        bl.setContentsMargins(0, 0, 0, 0)
        self.table = _stats_table()
        bl.addWidget(self.table, 1)
        info = QLabel(
            "<b>Como funciona:</b> o Monte Carlo sorteia o preço do ativo no vencimento num mundo "
            "\"neutro ao risco\" (rende os juros r) e calcula a média do que a opção pagaria, trazida a valor "
            "presente. Com mais simulações, a estimativa converge para a fórmula de Black-Scholes (linha "
            "tracejada). A faixa azul é o intervalo de confiança de 95%.<br><br><b>Vol. implícita:</b> informe o "
            "preço da opção no mercado para ver que volatilidade ele embute. Se a implícita for maior que a "
            "histórica, a opção está \"cara\" em relação ao que o ativo costuma oscilar.<br><br>Na B3, calls sobre "
            "ações são americanas; sem dividendos até o vencimento, valem o mesmo que a europeia.")
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignTop)
        info.setStyleSheet("color: #8b949e;")
        bl.addWidget(info, 1)
        vs.addWidget(bottom)
        vs.setSizes([480, 360])
        return vs

    def _on_expiry(self, *_) -> None:
        today = np.datetime64(date.today())
        exp = np.datetime64(self.expiry.date().toString("yyyy-MM-dd"))
        self.days.setValue(max(int(np.busday_count(today, exp)), 0))

    def _fetch_cdi(self) -> None:
        self.status.setText("Buscando CDI no Banco Central…")
        w = Worker(sources.current_cdi)
        w.signals.finished.connect(lambda out: (self.r.setValue(out[0] * 100),
                                                self.status.setText(f"CDI de {out[1]}: {out[0]:.2%} a.a.")))
        w.signals.error.connect(lambda m: self.status.setText(f"Não foi possível buscar o CDI: {m.splitlines()[0]}"))
        self.pool.start(w)

    def _fill_from_asset(self) -> None:
        t = self.ticker.text().strip()
        if not t:
            return
        self.settings.setValue("opt_ticker", t)
        years = self.years.value()
        start = str((pd.Timestamp.today() - pd.DateOffset(days=int(365.25 * (years + 0.5)))).date())
        self.status.setText(f"Carregando {t}…")
        w = Worker(sources.load_prices, t, self.src.currentText(), start)
        w.signals.finished.connect(lambda df: self._on_asset(df, t, years))
        w.signals.error.connect(lambda m: self.status.setText(f"Erro: {m.split(chr(10))[0]}"))
        self.pool.start(w)

    def _on_asset(self, df: pd.DataFrame, t: str, years: float) -> None:
        try:
            est = mc.estimate(df["close"], years)
        except ValueError as e:
            self.status.setText(str(e))
            return
        self.s.setValue(est.s0)
        if abs(self.k.value() - 30) < 1e-9:  # strike ainda no padrão: põe no dinheiro
            self.k.setValue(round(est.s0, 2))
        self.sigma.setValue(est.sigma_annual * 100)
        self.hist_vol = est.sigma_annual
        self.status.setText(f"{t.upper()}: preço {est.s0:,.2f} em {est.last_date:%d/%m/%Y}, volatilidade "
                            f"histórica {est.sigma_annual:.1%} a.a. ({years:g} anos).")

    def spec(self) -> op.OptionSpec:
        kind = "call" if self.kind.currentIndex() == 0 else "put"
        return op.OptionSpec(kind, self.s.value(), self.k.value(), self.days.value(), self.r.value() / 100,
                             self.sigma.value() / 100, self.q.value() / 100)

    def run(self) -> None:
        if self.r.value() == 0 and self.get_rf() > 0:
            self.r.setValue(self.get_rf() * 100)
        o = self.o = self.spec()
        if o.days <= 0:
            self.status.setText("O vencimento precisa ser no futuro (prazo > 0 dias úteis).")
            return
        self.bs = op.black_scholes(o)
        self.greeks = op.greeks(o)
        self.pricer = op.MCOptionPricer(o)
        steps = int(np.interp(self.speed.value(), [1, 10], [120, 12]))
        # lotes crescentes (geométricos): a convergência aparece bem no eixo logarítmico
        total_pairs = max(self.n.value() // 2, 1000)
        sizes = np.diff(np.unique(np.geomspace(500, total_pairs, steps).astype(int)), prepend=0)
        self.batches = iter(int(s) for s in sizes if s > 0)
        self.conv.clear()
        self.dist.clear()
        # PlotDataItem (respeita o eixo logarítmico) com traço fino: sem traço o preenchimento some
        self.band_lo = pg.PlotDataItem(pen=pg.mkPen(79, 157, 255, 90, width=1))
        self.band_hi = pg.PlotDataItem(pen=pg.mkPen(79, 157, 255, 90, width=1))
        self.conv.addItem(self.band_lo)
        self.conv.addItem(self.band_hi)
        self.conv.addItem(pg.FillBetweenItem(self.band_lo, self.band_hi, brush=pg.mkBrush(79, 157, 255, 60)))
        self.est_item = self.conv.plot(pen=pg.mkPen("#4f9dff", width=2), name="Monte Carlo")
        self.conv.plot([], [], pen=None, fillLevel=0, brush=pg.mkBrush(79, 157, 255, 60), name="IC 95%")
        self.y_span = None
        self.conv.addItem(pg.InfiniteLine(pos=self.bs, angle=0,
                                          pen=pg.mkPen(C_MEDIAN, width=1.5, style=Qt.DashLine)))
        self.conv.plot([], [], pen=pg.mkPen(C_MEDIAN, width=1.5, style=Qt.DashLine), name=f"Black-Scholes {self.bs:.4f}")
        self.dist_bars = pg.BarGraphItem(x=[0], height=[0], width=1)
        self.dist.addItem(self.dist_bars)
        self.dist.addItem(pg.InfiniteLine(pos=o.k, angle=90, pen=pg.mkPen(C_NOW, width=1.5),
                                          label=f"strike {o.k:,.2f}", labelOpts={"color": C_NOW, "position": 0.9}))
        self.btn_run.setEnabled(False)
        self.log(f"Opção {o.kind} S={o.s:.2f} K={o.k:.2f} {o.days} d.u. r={o.r:.2%} σ={o.sigma:.1%}: "
                 f"Black-Scholes {self.bs:.4f}")
        self.anim.start(_interval(self.speed))

    def _step(self) -> bool:
        try:
            size = next(self.batches)
        except StopIteration:
            return False
        res = self.pricer.step(size)
        h = np.array(res.history)
        self.est_item.setData(h[:, 0], h[:, 1])
        self.band_lo.setData(h[:, 0], h[:, 1] - 1.96 * h[:, 2])
        self.band_hi.setData(h[:, 0], h[:, 1] + 1.96 * h[:, 2])
        if self.y_span is None:  # escala fixa a partir do 1º lote: dá para ver o intervalo estreitar
            self.y_span = max(2.2 * 1.96 * h[0, 2], abs(self.bs) * 0.002, 1e-4)
        self.conv.setYRange(self.bs - self.y_span, self.bs + self.y_span)
        st = res.payoffs_sample
        if len(st):
            lo, hi = np.percentile(st, [0.5, 99.5])
            counts, edges = np.histogram(st, bins=50, range=(lo, hi))
            centers = (edges[:-1] + edges[1:]) / 2
            itm = centers > self.o.k if self.o.kind == "call" else centers < self.o.k
            brushes = [pg.mkBrush(46, 204, 113, 170) if x else pg.mkBrush(120, 126, 136, 140) for x in itm]
            self.dist_bars.setOpts(x=centers, height=counts / counts.sum() * 100,
                                   width=(edges[1] - edges[0]) * 0.9, brushes=brushes)
        self._fill(res, final=False)
        return True

    def _finish(self) -> None:
        self.btn_run.setEnabled(True)
        if self.pricer is None or self.pricer.n == 0:
            return
        res = self.pricer.result()
        self._fill(res, final=True)
        lo, hi = res.ci95
        self.status.setText(f"Preço justo ≈ {res.price:.4f} (Monte Carlo, IC 95% {lo:.4f} – {hi:.4f}); "
                            f"Black-Scholes {self.bs:.4f}.")

    def _fill(self, res: op.MCOptionResult, final: bool) -> None:
        o, g = self.o, self.greeks
        lo, hi = res.ci95
        rows = [
            ("Simulações", f"{res.n:,}" + ("" if final else " (parcial)")),
            ("Preço justo — Monte Carlo", f"{res.price:.4f}  (IC 95%: {lo:.4f} – {hi:.4f})"),
            ("Preço justo — Black-Scholes", f"{self.bs:.4f}"),
            ("Diferença MC − BS", f"{res.price - self.bs:+.4f}  ({res.stderr:.4f} de erro-padrão)"),
            ("Valor intrínseco", f"{max(o.s - o.k, 0) if o.kind == 'call' else max(o.k - o.s, 0):.4f}"),
            ("Chance de exercício (no vencimento)", _pct(g["prob_exercicio"], False)),
            ("Delta", f"{g['delta']:+.4f}  (por R$ 1 no ativo)"),
            ("Gamma", f"{g['gamma']:.5f}"),
            ("Vega", f"{g['vega']:+.4f}  (por 1 p.p. de volatilidade)"),
            ("Theta", f"{g['theta']:+.4f}  (por dia útil)"),
            ("Rho", f"{g['rho']:+.4f}  (por 1 p.p. de juros)"),
        ]
        if self.market.value() > 0:
            iv = op.implied_vol(o, self.market.value())
            if iv is None:
                rows.append(("Volatilidade implícita", "preço de mercado fora do intervalo possível"))
            else:
                ref = getattr(self, "hist_vol", None) or o.sigma
                verdict = "cara" if iv > ref * 1.05 else ("barata" if iv < ref * 0.95 else "em linha")
                rows.append(("Volatilidade implícita", f"{iv:.1%} a.a. — opção {verdict} vs. {ref:.1%} "
                             f"({'histórica' if getattr(self, 'hist_vol', None) else 'usada'})"))
                rows.append(("Mercado − preço justo", f"{self.market.value() - self.bs:+.4f}"))
        _fill_rows(self.table, rows)


# ================================================================ valuation
class ValuationTab(QWidget):
    def __init__(self, log, get_rf, parent=None) -> None:
        super().__init__(parent)
        self.log, self.get_rf = log, get_rf
        self.settings = app_settings()
        self.pool = QThreadPool.globalInstance()
        self.dist: np.ndarray | None = None
        self.anim = _Animator(self._step, self._finish)

        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._controls())
        split.addWidget(self._results())
        split.setStretchFactor(1, 1)
        split.setSizes([380, 1060])

    def _controls(self) -> QWidget:
        p = _Panel()
        gl = p.group("Empresa")
        self.ticker = QLineEdit(self.settings.value("val_ticker", "PETR4"))
        self.btn_fetch = QPushButton("Buscar dados")
        self.btn_fetch.setToolTip("Preenche os campos com dados do Yahoo Finance (confira!)")
        self.btn_fetch.clicked.connect(self._fetch)
        p.put(gl, 0, 0, "Ação", self.ticker)
        gl.addWidget(self.btn_fetch, 0, 2, 1, 2)

        gl = p.group("Dados por ação (editáveis)")
        self.price = _dspin(0, 1e9, 0, 2, tip="Preço atual")
        self.eps = _dspin(-1e6, 1e6, 0, 3, tip="LPA: lucro por ação dos últimos 12 meses")
        self.bvps = _dspin(-1e6, 1e6, 0, 3, tip="VPA: patrimônio líquido por ação")
        self.dps12 = _dspin(0, 1e6, 0, 3, tip="Dividendos + JCP por ação pagos nos últimos 12 meses")
        self.dps5 = _dspin(0, 1e6, 0, 3, tip="Média anual de dividendos + JCP dos últimos 5 anos completos")
        self.fcf = _dspin(-1e6, 1e6, 0, 3, tip="Fluxo de caixa livre por ação (12 meses)")
        p.put(gl, 0, 0, "Preço", self.price)
        p.put(gl, 0, 1, "LPA", self.eps)
        p.put(gl, 1, 0, "VPA", self.bvps)
        p.put(gl, 1, 1, "FCF/ação", self.fcf)
        p.put(gl, 2, 0, "Div. 12m", self.dps12)
        p.put(gl, 2, 1, "Div. méd. 5a", self.dps5)

        gl = p.group("Premissas")
        self.mult = _dspin(1, 100, 22.5, 1, tip="Graham: 22,5 = P/L 15 × P/VPA 1,5")
        self.yld = _dspin(0.5, 50, 6, 1, " %", tip="Bazin: dividend yield mínimo desejado")
        self.use_avg = QCheckBox("Bazin/Gordon com média de 5 anos")
        self.use_avg.setChecked(True)
        self.use_avg.setToolTip("Mais estável que só os últimos 12 meses (que podem ter dividendos extraordinários)")
        self.k = _dspin(0, 100, 0, 1, " % a.a.", 0.5, "Custo de capital / taxa de desconto (ex.: CDI + prêmio de risco)")
        self.btn_k = QPushButton("CDI + 4%")
        self.btn_k.setToolTip("Usa o CDI atual (Banco Central) + 4 p.p. de prêmio de risco")
        self.btn_k.clicked.connect(self._k_from_cdi)
        self.g_perp = _dspin(-10, 50, 4, 1, " % a.a.", 0.5, "Crescimento perpétuo (deve ficar abaixo do desconto)")
        self.g_years = _dspin(-50, 100, 8, 1, " % a.a.", 1, "Crescimento do fluxo de caixa nos primeiros anos")
        self.n_years = _ispin(1, 30, 5, " anos", tip="Anos de crescimento antes da perpetuidade")
        p.put(gl, 0, 0, "Graham ×", self.mult)
        p.put(gl, 0, 1, "Bazin yield", self.yld)
        gl.addWidget(self.use_avg, 1, 0, 1, 4)
        p.put(gl, 2, 0, "Desconto (k)", self.k)
        gl.addWidget(self.btn_k, 2, 2, 1, 2)
        p.put(gl, 3, 0, "Cresc. anos", self.g_years)
        p.put(gl, 3, 1, "Por", self.n_years)
        p.put(gl, 4, 0, "Cresc. perp.", self.g_perp)

        gl = p.group("Incerteza do FCD (Monte Carlo)")
        self.sd_g = _dspin(0, 50, 3, 1, " p.p.", tip="Desvio-padrão do crescimento dos primeiros anos")
        self.sd_gp = _dspin(0, 20, 1, 1, " p.p.", tip="Desvio-padrão do crescimento perpétuo")
        self.sd_k = _dspin(0, 20, 1.5, 1, " p.p.", tip="Desvio-padrão da taxa de desconto")
        self.n_mc = _ispin(1000, 500_000, 50_000, step=10_000)
        self.speed = _speed_slider()
        p.put(gl, 0, 0, "± cresc.", self.sd_g)
        p.put(gl, 0, 1, "± perp.", self.sd_gp)
        p.put(gl, 1, 0, "± desconto", self.sd_k)
        p.put(gl, 1, 1, "Simulações", self.n_mc)
        p.put(gl, 2, 0, "Velocidade", self.speed, span=3)

        self.btn_run = _run_button("▶  Calcular preço justo")
        self.btn_run.clicked.connect(self.run)
        self.status = _status()
        return p.finish([self.btn_run, self.status])

    def _results(self) -> QWidget:
        vs = QSplitter(Qt.Vertical)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Método", "Preço justo", "vs. preço atual", "Conta", "Observação"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        vs.addWidget(self.table)
        glw = pg.GraphicsLayoutWidget()
        self.bar_plot = glw.addPlot(row=0, col=0)
        self.mc_plot = glw.addPlot(row=0, col=1)
        self.bar_plot.setTitle("Preço justo por método × preço atual", size="10pt")
        self.bar_plot.showGrid(x=True, alpha=0.15)
        self.mc_plot.setTitle("Fluxo de caixa descontado com incerteza (Monte Carlo)", size="10pt")
        self.mc_plot.showGrid(x=True, y=True, alpha=0.1)
        self.mc_plot.setLabel("left", "% das simulações")
        vs.addWidget(glw)
        self.notes = QLabel(
            "Os dados vêm do Yahoo Finance e podem estar desatualizados ou em outra base (units, ações "
            "preferenciais). Confira no RI da empresa. Preço justo depende muito das premissas: use as faixas "
            "do Monte Carlo como ordem de grandeza, não como número exato.")
        self.notes.setWordWrap(True)
        self.notes.setStyleSheet("color: #8b949e;")
        vs.addWidget(self.notes)
        vs.setSizes([210, 480, 70])
        return vs

    def _fetch(self) -> None:
        t = self.ticker.text().strip().upper()
        if not t:
            return
        self.settings.setValue("val_ticker", t)
        self.btn_fetch.setEnabled(False)
        self.status.setText(f"Buscando dados de {t}…")

        def job():
            f = va.fetch_fundamentals(t)
            try:  # preço oficial da B3, se o ativo estiver na base
                f.price = float(sources.load_prices(t, sources.SRC_B3)["close"].iloc[-1])
                f.source += " · preço da Base B3"
            except Exception:  # noqa: BLE001
                pass
            return f

        w = Worker(job)
        w.signals.finished.connect(self._on_fetched)
        w.signals.error.connect(lambda m: (self.btn_fetch.setEnabled(True),
                                           self.status.setText(f"Erro: {m.splitlines()[0]}")))
        self.pool.start(w)

    def _on_fetched(self, f: va.Fundamentals) -> None:
        self.btn_fetch.setEnabled(True)
        for w, v in ((self.price, f.price), (self.eps, f.eps), (self.bvps, f.bvps), (self.dps12, f.dps_12m),
                     (self.dps5, f.dps_avg5), (self.fcf, f.fcf_ps)):
            w.setValue(v)
        msg = f"Dados de {f.source}."
        if f.warnings:
            msg += " " + " ".join(f.warnings)
        self.status.setText(msg)
        self.log(f"Valuation: dados de {f.ticker} carregados ({f.source})")
        if self.k.value() == 0:
            self._k_from_cdi()

    def _k_from_cdi(self) -> None:
        w = Worker(sources.current_cdi)
        w.signals.finished.connect(lambda out: (self.k.setValue(out[0] * 100 + 4),
                                                self.status.setText(self.status.text() +
                                                                    f" Desconto = CDI {out[0]:.2%} + 4 p.p.")))
        w.signals.error.connect(lambda m: self.status.setText("Não foi possível buscar o CDI; informe o desconto."))
        self.pool.start(w)

    def fundamentals(self) -> va.Fundamentals:
        return va.Fundamentals(ticker=self.ticker.text().strip().upper(), price=self.price.value(),
                               eps=self.eps.value(), bvps=self.bvps.value(), dps_12m=self.dps12.value(),
                               dps_avg5=self.dps5.value(), fcf_ps=self.fcf.value())

    def params(self) -> va.ValuationParams:
        return va.ValuationParams(graham_mult=self.mult.value(), bazin_yield=self.yld.value() / 100,
                                  bazin_use_avg=self.use_avg.isChecked(), k=self.k.value() / 100,
                                  g_perp=self.g_perp.value() / 100, g_years=self.g_years.value() / 100,
                                  years=self.n_years.value())

    def run(self) -> None:
        f, p = self.fundamentals(), self.params()
        if f.price <= 0:
            self.status.setText("Informe o preço atual (ou clique em Buscar dados).")
            return
        if p.k <= 0:
            self.status.setText("Informe a taxa de desconto (ou clique em CDI + 4%).")
            return
        self.f, self.p = f, p
        self.res = va.evaluate(f, p)
        self._fill_table()
        self._draw_bars()
        # FCD com incerteza: histograma se formando em lotes
        self.mc_plot.clear()
        self.dist = None
        if f.fcf_ps <= 0:
            self.status.setText("Sem fluxo de caixa livre positivo: o FCD (e seu Monte Carlo) não se aplica.")
            return
        self.dist = va.dcf_monte_carlo(f.fcf_ps, p.g_years, p.years, p.g_perp, p.k, self.sd_g.value() / 100,
                                       self.sd_gp.value() / 100, self.sd_k.value() / 100, n=self.n_mc.value())
        if len(self.dist) == 0:
            self.status.setText("As premissas não geraram valores válidos (desconto ≤ crescimento perpétuo).")
            return
        hi_cap = np.percentile(self.dist, 99)
        self.range = (max(0.0, float(np.percentile(self.dist, 0.5))), float(hi_cap))
        steps = int(np.interp(self.speed.value(), [1, 10], [60, 6]))
        self.cuts = iter(np.linspace(0, len(self.dist), steps + 1).astype(int)[1:])
        self.mc_bars = pg.BarGraphItem(x=[0], height=[0], width=1, brush=pg.mkBrush(79, 157, 255, 170))
        self.mc_plot.addItem(self.mc_bars)
        self.mc_plot.addItem(pg.InfiniteLine(pos=f.price, angle=90, pen=pg.mkPen(C_NOW, width=2),
                                             label=f"preço atual {f.price:,.2f}",
                                             labelOpts={"color": C_NOW, "position": 0.92}))
        self.btn_run.setEnabled(False)
        self.anim.start(_interval(self.speed))

    def _fill_table(self) -> None:
        t = self.table
        t.setRowCount(len(self.res))
        for r, row in enumerate(self.res.itertuples()):
            fair, m = row.preco_justo, row.margem
            cells = [QTableWidgetItem(row.metodo),
                     NumItem("não se aplica" if fair is None else f"{fair:,.2f}", fair),
                     NumItem("—" if m is None else f"{m:+.1%}", m),
                     QTableWidgetItem(row.conta), QTableWidgetItem(row.obs)]
            if m is not None:
                cells[2].setForeground(QColor(C_BUY if m > 0 else C_SELL))
                cells[2].setToolTip("Positivo: preço justo acima do atual (ação parece barata por este método)")
            for c, item in enumerate(cells):
                t.setItem(r, c, item)
        t.resizeColumnsToContents()

    def _draw_bars(self) -> None:
        self.bar_plot.clear()
        ok = self.res.dropna(subset=["preco_justo"]).reset_index(drop=True)
        if ok.empty:
            return
        y = np.arange(len(ok))
        colors = [pg.mkBrush(46, 204, 113, 170) if v > self.f.price else pg.mkBrush(255, 92, 92, 170)
                  for v in ok["preco_justo"]]
        self.bar_plot.addItem(pg.BarGraphItem(x0=0, y=y, width=ok["preco_justo"].to_numpy(), height=0.6,
                                              brushes=colors))
        self.bar_plot.addItem(pg.InfiniteLine(pos=self.f.price, angle=90, pen=pg.mkPen(C_NOW, width=2),
                                              label=f"atual {self.f.price:,.2f}",
                                              labelOpts={"color": C_NOW, "position": 0.95}))
        self.bar_plot.getAxis("left").setTicks([[(i, m) for i, m in enumerate(ok["metodo"])]])
        self.bar_plot.setLimits(yMin=-1, yMax=len(ok))

    def _step(self) -> bool:
        try:
            cut = next(self.cuts)
        except StopIteration:
            return False
        part = self.dist[:cut]
        counts, edges = np.histogram(part, bins=60, range=self.range)
        centers = (edges[:-1] + edges[1:]) / 2
        self.mc_bars.setOpts(x=centers, height=counts / max(len(part), 1) * 100, width=(edges[1] - edges[0]) * 0.9)
        self._mc_summary(part, final=False)
        return True

    def _finish(self) -> None:
        self.btn_run.setEnabled(True)
        if self.dist is None or not len(self.dist):
            return
        d = self.dist
        for pct, style in ((5, Qt.DotLine), (50, Qt.DashLine), (95, Qt.DotLine)):
            v = float(np.percentile(d, pct))
            self.mc_plot.addItem(pg.InfiniteLine(pos=v, angle=90, pen=pg.mkPen(C_MEDIAN, width=1.5, style=style),
                                                 label=f"P{pct} {v:,.2f}",
                                                 labelOpts={"color": C_MEDIAN, "position": 0.75 if pct == 50 else 0.6}))
        self._mc_summary(d, final=True)

    def _mc_summary(self, d: np.ndarray, final: bool) -> None:
        if not len(d):
            return
        p5, p50, p95 = np.percentile(d, [5, 50, 95])
        above = float((d > self.f.price).mean())
        txt = (f"FCD com incerteza ({len(d):,} cenários{'' if final else ', parcial'}): mediana {p50:,.2f}, "
               f"90% entre {p5:,.2f} e {p95:,.2f}. Em {above:.0%} dos cenários o preço justo fica acima do "
               f"preço atual ({self.f.price:,.2f}).")
        self.status.setText(txt)
        if final:
            self.log("Valuation " + self.f.ticker + ": " + txt)


# ================================================================ seção
class SimulationSection(QTabWidget):
    def __init__(self, log, get_rf, parent=None) -> None:
        super().__init__(parent)
        self.mc_tab = MonteCarloTab(log, get_rf)
        self.opt_tab = OptionsTab(log, get_rf)
        self.val_tab = ValuationTab(log, get_rf)
        self.addTab(self.mc_tab, "Monte Carlo (preço futuro)")
        self.addTab(self.opt_tab, "Opções (preço justo)")
        self.addTab(self.val_tab, "Valuation (preço justo)")
        # CDI atual (Banco Central) como padrão dos juros nas três abas
        w = Worker(sources.current_cdi)
        w.signals.finished.connect(self._on_cdi)
        QThreadPool.globalInstance().start(w)

    def _on_cdi(self, out: tuple[float, str]) -> None:
        cdi, when = out
        if self.mc_tab.rf.value() == 0:
            self.mc_tab.rf.setValue(cdi * 100)
        if self.opt_tab.r.value() == 0:
            self.opt_tab.r.setValue(cdi * 100)
        if self.val_tab.k.value() == 0:
            self.val_tab.k.setValue(cdi * 100 + 4)
        for tab in (self.mc_tab, self.opt_tab, self.val_tab):
            tab.cdi_info = f"CDI de {when}: {cdi:.2%} a.a."
