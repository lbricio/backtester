"""Sub-aba Varredura: várias estratégias × vários ativos."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QDate, Qt, QThreadPool, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit, QProgressBar, QPushButton,
    QSpinBox, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from quantbt import b3, external, sources
from quantbt.scan import HEATMAP_METRICS, pivot, run_scan, summarize_assets, summarize_strategies
from quantbt.settings import app_settings
from quantbt.gui.widgets import C_BUY, C_SELL, NumItem, Worker

KINDS: dict[str, tuple[str, ...] | None] = {"Todos": None, **b3.KIND_CODES}
DEFAULT_ASSETS = ("PETR4 VALE3 ITUB4 BBDC4 BBAS3 ITSA4 B3SA3 ABEV3 WEGE3 ELET3 SUZB3 GGBR4 PRIO3 RENT3 "
                  "BOVA11 IVVB11")


def _cmap() -> pg.ColorMap:
    return pg.ColorMap(np.array([0.0, 0.5, 1.0]),
                       np.array([[214, 69, 65, 255], [30, 34, 41, 255], [46, 160, 255, 255]], dtype=np.ubyte))


def _fmt(v, kind: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:+.1%}" if kind == "pct" else (f"{v:.0%}" if kind == "share" else f"{v:+.2f}")


class RectHeatmap(pg.PlotWidget):
    """Mapa de calor ativo (linhas) × estratégia (colunas)."""

    hovered = Signal(str)
    clicked = Signal(str, str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.img = pg.ImageItem()
        self.addItem(self.img)
        self.img.setColorMap(_cmap())
        self.invertY(True)
        self.hideButtons()
        self.getPlotItem().setMenuEnabled(False)
        self.rows: list[str] = []
        self.cols: list[str] = []
        self.values: np.ndarray | None = None
        self.fmt = "num"
        self.bar = pg.ColorBarItem(values=(-1, 1), colorMap=_cmap(), interactive=False, width=12)
        self.bar.setImageItem(self.img, insert_in=self.getPlotItem())
        self.scene().sigMouseMoved.connect(self._move)
        self.scene().sigMouseClicked.connect(self._click)

    def set_data(self, table: pd.DataFrame, fmt: str, center: float | None) -> None:
        self.rows, self.cols = table.index.tolist(), table.columns.tolist()
        self.values, self.fmt = table.to_numpy(dtype=float), fmt
        finite = self.values[np.isfinite(self.values)]
        if center is not None and len(finite):
            span = float(np.nanpercentile(np.abs(finite - center), 95)) or 1e-6
            lo, hi = center - span, center + span
        elif len(finite):
            lo, hi = float(np.nanpercentile(finite, 5)), float(np.nanpercentile(finite, 95))
            hi = hi if hi > lo else lo + 1e-6
        else:
            lo, hi = -1.0, 1.0
        fill = (lo + hi) / 2
        self.img.setImage(np.nan_to_num(self.values.T, nan=fill), levels=(lo, hi))
        self.bar.setLevels((lo, hi))
        step = max(1, -(-len(self.rows) // 30))
        self.getAxis("left").setTicks([[(i + 0.5, r) for i, r in enumerate(self.rows)][::step], []])
        short = [c if len(c) <= 14 else c[:13] + "…" for c in self.cols]
        self.getAxis("bottom").setTicks([[(j + 0.5, s) for j, s in enumerate(short)], []])
        self.setRange(xRange=(0, len(self.cols)), yRange=(0, len(self.rows)), padding=0.01)

    def _cell(self, pos):
        if self.values is None:
            return None
        p = self.getPlotItem().vb.mapSceneToView(pos)
        i, j = int(np.floor(p.y())), int(np.floor(p.x()))
        if 0 <= i < len(self.rows) and 0 <= j < len(self.cols):
            return i, j
        return None

    def _move(self, pos) -> None:
        c = self._cell(pos)
        if c:
            i, j = c
            self.hovered.emit(f"{self.rows[i]} × {self.cols[j]}: {_fmt(self.values[i, j], self.fmt)}")

    def _click(self, ev) -> None:
        c = self._cell(ev.scenePos())
        if c:
            self.clicked.emit(self.rows[c[0]], self.cols[c[1]])


def _table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.setSortingEnabled(True)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _fill(t: QTableWidget, rows: list[list[tuple]]) -> None:
    """rows: lista de células (texto, valor numérico ou None, cor ou None)."""
    t.setSortingEnabled(False)
    t.setRowCount(len(rows))
    for r, cells in enumerate(rows):
        for c, (text, value, color) in enumerate(cells):
            item = NumItem(text, value) if value is not None else QTableWidgetItem(text)
            if color:
                item.setForeground(QColor(color))
            t.setItem(r, c, item)
    t.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
    t.setSortingEnabled(True)


def _sign_color(v) -> str | None:
    if v is None or not np.isfinite(v):
        return None
    return C_BUY if v > 0 else C_SELL


class ScanTab(QWidget):
    """Precisa da janela principal para: custos/config, estratégias e abrir um par no backtest."""

    log = Signal(str)
    open_pair = Signal(str, str)  # ativo, estratégia

    def __init__(self, get_config, get_strategies, get_current, parent=None) -> None:
        super().__init__(parent)
        self.get_config, self.get_strategies, self.get_current = get_config, get_strategies, get_current
        self.settings = app_settings()
        self.pool = QThreadPool.globalInstance()
        self.worker: Worker | None = None
        self.results: pd.DataFrame | None = None
        self.names: dict[str, str] = {}

        root = QVBoxLayout(self)
        split = QSplitter(Qt.Vertical)
        root.addWidget(split)
        split.addWidget(self._controls())
        split.addWidget(self._results())
        split.setSizes([330, 530])
        self.refresh_strategies()

    # ------------------------------------------------------------ layout
    def _controls(self) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)

        # --- ativos
        g = QGroupBox("Ativos")
        gl = QGridLayout(g)
        self.assets = QPlainTextEdit(self.settings.value("scan_assets", DEFAULT_ASSETS))
        self.assets.setPlaceholderText("Códigos da B3 ou externos (BTC, DOLAR...), separados por espaço")
        self.assets.setMinimumHeight(70)
        self.kind = QComboBox()
        self.kind.addItems(list(KINDS))
        self.kind.setCurrentText("Ações / units")
        self.top_n = QSpinBox()
        self.top_n.setRange(0, 100_000)
        self.top_n.setValue(50)
        self.top_n.setSpecialValueText("todos")
        self.min_vol = QDoubleSpinBox()
        self.min_vol.setRange(0, 1e9)
        self.min_vol.setDecimals(0)
        self.min_vol.setSuffix(" mil")
        self.min_vol.setValue(500)
        self.min_vol.setToolTip("Volume financeiro médio mínimo por pregão (milhares de R$)")
        self.min_days = QSpinBox()
        self.min_days.setRange(0, 100)
        self.min_days.setSuffix(" %")
        self.min_days.setValue(90)
        self.since = QCheckBox("Incluir quem estreou depois do início")
        self.since.setChecked(True)
        btn_uni = QPushButton("Carregar universo da B3")
        btn_uni.clicked.connect(self._load_universe)
        self.preset = QComboBox()
        self.preset.addItems(list(external.PRESETS))
        btn_ext = QPushButton("+ Externos")
        btn_ext.clicked.connect(self._add_preset)
        gl.addWidget(self.assets, 0, 0, 1, 6)
        for col, (lbl, wd) in enumerate((("Tipo", self.kind), ("Máx.", self.top_n), ("Vol./dia", self.min_vol))):
            gl.addWidget(QLabel(lbl), 1, col * 2)
            gl.addWidget(wd, 1, col * 2 + 1)
        lbl_days = QLabel("Dias c/ negócio ≥")
        lbl_days.setToolTip("Frequência mínima de negociação: % dos pregões em que o ativo teve pelo menos um negócio (desde a estreia, se 'Incluir quem estreou' estiver ligado). 90% = negociou em 9 de cada 10 pregões. Descarta ativos que passam dias sem negociar, cujo preço fica parado e depois salta.")
        self.min_days.setToolTip("Frequência mínima de negociação: % dos pregões em que o ativo teve pelo menos um negócio (desde a estreia, se 'Incluir quem estreou' estiver ligado). 90% = negociou em 9 de cada 10 pregões. Descarta ativos que passam dias sem negociar, cujo preço fica parado e depois salta.")
        gl.addWidget(lbl_days, 2, 0)
        gl.addWidget(self.min_days, 2, 1)
        gl.addWidget(self.since, 2, 2, 1, 4)
        gl.addWidget(btn_uni, 3, 0, 1, 3)
        gl.addWidget(self.preset, 3, 3, 1, 2)
        gl.addWidget(btn_ext, 3, 5)
        h.addWidget(g, 3)

        # --- estratégias
        g = QGroupBox("Estratégias")
        v = QVBoxLayout(g)
        self.strat_list = QListWidget()
        self.strat_list.setToolTip("Marque as estratégias a testar")
        self.strat_list.setMinimumHeight(150)
        self.strat_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.strat_list.setTextElideMode(Qt.ElideRight)
        hb = QHBoxLayout()
        for text, state in (("Todas", True), ("Nenhuma", False)):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, s=state: self._check_all(s))
            hb.addWidget(b)
        note = QLabel("Cada estratégia usa os parâmetros padrão; a que está selecionada no painel usa os valores "
                      "do painel. Custos, slippage, vendido e CDI vêm do painel lateral.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #8b949e;")
        v.addWidget(self.strat_list, 1)
        v.addLayout(hb)
        v.addWidget(note)
        h.addWidget(g, 2)

        # --- período e execução
        g = QGroupBox("Período e execução")
        gl = QGridLayout(g)
        self.d_start = QDateEdit(QDate.currentDate().addYears(-10))
        self.d_end = QDateEdit(QDate.currentDate())
        for d in (self.d_start, self.d_end):
            d.setCalendarPopup(True)
            d.setDisplayFormat("dd/MM/yyyy")
        self.metric = QComboBox()
        for k, (label, _, _) in HEATMAP_METRICS.items():
            self.metric.addItem(label, k)
        self.metric.currentIndexChanged.connect(self._draw_heatmap)
        self.btn_run = QPushButton("▶  Rodar varredura")
        self.btn_run.setMinimumHeight(36)
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self.run)
        self.btn_stop = QPushButton("■ Parar")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        self.progress = QProgressBar()
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #8b949e;")
        gl.addWidget(QLabel("Início"), 0, 0)
        gl.addWidget(self.d_start, 0, 1)
        gl.addWidget(QLabel("Fim"), 1, 0)
        gl.addWidget(self.d_end, 1, 1)
        gl.addWidget(QLabel("Mapa"), 2, 0)
        gl.addWidget(self.metric, 2, 1)
        gl.addWidget(self.btn_run, 3, 0, 1, 2)
        gl.addWidget(self.btn_stop, 4, 0, 1, 2)
        gl.addWidget(self.progress, 5, 0, 1, 2)
        gl.addWidget(self.status, 6, 0, 1, 2)
        gl.setRowStretch(7, 1)
        h.addWidget(g, 2)
        return w

    def _results(self) -> QWidget:
        self.tabs = QTabWidget()
        # mapa de calor
        hm = QWidget()
        hv = QVBoxLayout(hm)
        hv.setContentsMargins(0, 0, 0, 0)
        self.heatmap = RectHeatmap()
        self.hover = QLabel("Passe o mouse no mapa; clique numa célula para abrir o par no backtest.")
        self.hover.setStyleSheet("color: #8b949e;")
        self.heatmap.hovered.connect(self.hover.setText)
        self.heatmap.clicked.connect(self.open_pair.emit)
        hv.addWidget(self.heatmap, 1)
        hv.addWidget(self.hover)
        self.tabs.addTab(hm, "Mapa de calor")
        # rankings
        self.t_strat = _table(["Estratégia", "Ativos", "Vence o CDI", "Vence o B&H", "Ganho s/ CDI (mediana)",
                               "Sharpe (mediana)", "CAGR (mediana)", "Drawdown (mediana)", "Pior drawdown",
                               "Operações/ano", "Tempo posicionado"])
        self.tabs.addTab(self.t_strat, "Ranking de estratégias")
        self.t_asset = _table(["Ativo", "Nome", "Estratégias", "Vencem o CDI", "Melhor estratégia",
                               "Ganho da melhor s/ CDI", "Sharpe da melhor", "Buy & Hold (a.a.)", "CDI (a.a.)", "Anos"])
        self.t_asset.cellDoubleClicked.connect(self._open_from_assets)
        self.tabs.addTab(self.t_asset, "Ranking de ativos")
        self.t_all = _table(["Ativo", "Estratégia", "Ganho s/ CDI", "Ganho s/ B&H", "CAGR", "Sharpe", "Drawdown",
                             "Operações", "Op./ano", "Tempo posic.", "Anos", "Erro"])
        self.t_all.cellDoubleClicked.connect(self._open_from_all)
        self.tabs.addTab(self.t_all, "Todos os resultados")
        box = QWidget()
        bv = QVBoxLayout(box)
        bv.setContentsMargins(0, 0, 0, 0)
        bv.addWidget(self.tabs, 1)
        hb = QHBoxLayout()
        expl = QLabel("<b>Como ler:</b> \"Vence o CDI\" é a fração dos ativos em que a estratégia rendeu mais que o "
                      "CDI acumulado nas mesmas datas (Banco Central). Prefira estratégias que vencem em muitos "
                      "ativos com mediana positiva; um único resultado brilhante costuma ser sorte. Dois cliques "
                      "numa linha (ou clique no mapa) abrem o par no backtest.")
        expl.setWordWrap(True)
        expl.setStyleSheet("color: #8b949e;")
        btn_exp = QPushButton("Exportar resultados…")
        btn_exp.clicked.connect(self._export)
        hb.addWidget(expl, 1)
        hb.addWidget(btn_exp)
        bv.addLayout(hb)
        return box

    # ------------------------------------------------------------ seleção
    def refresh_strategies(self) -> None:
        """Atualiza a lista (ex.: estratégia nova do editor), mantendo o que já estava marcado."""
        prev = {self.strat_list.item(i).text(): self.strat_list.item(i).checkState()
                for i in range(self.strat_list.count())}
        self.strat_list.clear()
        for s in self.get_strategies():
            it = QListWidgetItem(s.name)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(prev.get(s.name, Qt.Checked))  # estratégias novas entram marcadas
            self.strat_list.addItem(it)

    def _check_all(self, on: bool) -> None:
        for i in range(self.strat_list.count()):
            self.strat_list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _selected_strategies(self):
        names = {self.strat_list.item(i).text() for i in range(self.strat_list.count())
                 if self.strat_list.item(i).checkState() == Qt.Checked}
        return [s for s in self.get_strategies() if s.name in names]

    def _codes(self) -> list[str]:
        raw = [t for t in re.split(r"[\s,;]+", self.assets.toPlainText().upper()) if t]
        return list(dict.fromkeys(raw))

    def _dates(self) -> tuple[str, str]:
        return self.d_start.date().toString("yyyy-MM-dd"), self.d_end.date().toString("yyyy-MM-dd")

    def _load_universe(self) -> None:
        start, end = self._dates()
        self.status.setText("Calculando o universo…")
        w = Worker(b3.liquid_universe, self.top_n.value(), start, end, self.min_days.value() / 100,
                   KINDS[self.kind.currentText()], self.min_vol.value() * 1000, self.since.isChecked(), 252)
        w.signals.finished.connect(self._on_universe)
        w.signals.error.connect(lambda m: self.status.setText(f"Erro: {m.splitlines()[0]}"))
        self.pool.start(w)

    def _on_universe(self, df: pd.DataFrame) -> None:
        if df.empty:
            self.status.setText("Nenhum ativo passou nos filtros.")
            return
        keep_ext = [c for c in self._codes() if external.is_external(c)]
        self.assets.setPlainText(" ".join(list(df["ticker"]) + keep_ext))
        self.names = dict(zip(df["ticker"], df["name"]))
        self.status.setText(f"{len(df)} ativos carregados" + (f" + {len(keep_ext)} externos" if keep_ext else "") + ".")

    def _add_preset(self) -> None:
        cur = self._codes()
        add = [c for c in external.PRESETS[self.preset.currentText()] if c not in cur]
        self.assets.setPlainText(" ".join(cur + add))

    # ------------------------------------------------------------ execução
    def run(self) -> None:
        codes, strats = self._codes(), self._selected_strategies()
        if not codes or not strats:
            self.status.setText("Escolha ao menos um ativo e uma estratégia.")
            return
        self.settings.setValue("scan_assets", " ".join(codes))
        start, end = self._dates()
        cfg = self.get_config()  # lido aqui, na thread da interface
        cur, cur_params = self.get_current()
        params = {s.name: (cur_params if s is cur else s.defaults()) for s in strats}
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress.setValue(0)
        self.status.setText(f"Carregando {len(codes)} ativos…")
        worker: Worker = None  # type: ignore[assignment]

        def job():
            data, missing = sources.load_many(codes, start, end)
            try:
                cdi = sources.cdi_index(start)
                note = ""
            except Exception:  # noqa: BLE001
                cdi, note = None, f"CDI histórico indisponível: usando {cfg.risk_free:.2%} a.a. constante."
            res = run_scan(data, strats, cfg, params, cdi,
                           progress=lambda i, n: worker.signals.progress.emit(i, n),
                           should_stop=lambda: worker.cancelled)
            return res, missing, note

        worker = Worker(job)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_done)
        worker.signals.error.connect(self._on_error)
        self.worker = worker
        self.log.emit(f"Varredura: {len(strats)} estratégias × {len(codes)} ativos ({start} → {end})")
        self.pool.start(worker)

    def _stop(self) -> None:
        if self.worker:
            self.worker.cancelled = True

    def _on_progress(self, i: int, n: int) -> None:
        self.progress.setMaximum(n)
        self.progress.setValue(i)
        self.status.setText(f"Rodando backtests… {i:,} de {n:,}")

    def _on_error(self, msg: str) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        short = msg.split("\n\nTraceback")[0]
        self.status.setText(f"Erro: {short}")
        self.log.emit(f"ERRO — Varredura: {short}")

    def _on_done(self, out) -> None:
        res, missing, note = out
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.results = res
        ok = res[res["error"].fillna("") == ""]
        self._draw_heatmap()
        self._fill_strategies()
        self._fill_assets()
        self._fill_all()
        n_err = len(res) - len(ok)
        msg = f"Pronto: {len(ok):,} backtests ({ok['ticker'].nunique()} ativos × {ok['strategy'].nunique()} estratégias)."
        if n_err:
            msg += f" {n_err} combinações com erro (veja 'Todos os resultados')."
        if missing:
            msg += f" Não encontrados: {', '.join(missing[:8])}{'…' if len(missing) > 8 else ''}."
        if note:
            msg += " " + note
        if self.worker and self.worker.cancelled:
            msg = "Interrompida. " + msg
        self.status.setText(msg)
        self.log.emit("Varredura: " + msg)

    # ------------------------------------------------------------ resultados
    def _draw_heatmap(self, *_) -> None:
        if self.results is None:
            return
        key = self.metric.currentData()
        label, fmt, center = HEATMAP_METRICS[key]
        table = pivot(self.results, key)
        if table.empty:
            return
        # ativos ordenados pelo melhor resultado, para os destaques ficarem no topo
        table = table.loc[table.max(axis=1).sort_values(ascending=False).index]
        self.heatmap.set_data(table, fmt, center)
        self.heatmap.setTitle(f"{label} — ativo × estratégia", size="10pt")

    def _fill_strategies(self) -> None:
        s = summarize_strategies(self.results)
        rows = []
        for r in s.itertuples():
            rows.append([
                (r.strategy, None, None), (str(r.ativos), r.ativos, None),
                (_fmt(r.vence_cdi, "share"), r.vence_cdi, C_BUY if r.vence_cdi >= 0.5 else C_SELL),
                (_fmt(r.vence_bh, "share"), r.vence_bh,
                 None if pd.isna(r.vence_bh) else (C_BUY if r.vence_bh >= 0.5 else C_SELL)),
                (_fmt(r.excesso_cdi_mediano, "pct"), r.excesso_cdi_mediano, _sign_color(r.excesso_cdi_mediano)),
                (_fmt(r.sharpe_mediano, "num"), r.sharpe_mediano, _sign_color(r.sharpe_mediano)),
                (_fmt(r.cagr_mediano, "pct"), r.cagr_mediano, None),
                (_fmt(r.dd_mediano, "pct"), r.dd_mediano, None), (_fmt(r.pior_dd, "pct"), r.pior_dd, None),
                (f"{r.trades_ano:.1f}", r.trades_ano, None), (_fmt(r.exposicao, "share"), r.exposicao, None),
            ])
        _fill(self.t_strat, rows)

    def _fill_assets(self) -> None:
        a = summarize_assets(self.results)
        rows = []
        for r in a.itertuples():
            name = self.names.get(r.ticker) or external.describe(r.ticker)
            rows.append([
                (r.ticker, None, None), (name, None, None), (str(r.estrategias), r.estrategias, None),
                (_fmt(r.vencem_cdi, "share"), r.vencem_cdi, C_BUY if r.vencem_cdi >= 0.5 else None),
                (r.melhor, None, None),
                (_fmt(r.melhor_excesso, "pct"), r.melhor_excesso, _sign_color(r.melhor_excesso)),
                (_fmt(r.melhor_sharpe, "num"), r.melhor_sharpe, None),
                (_fmt(r.bh_cagr, "pct"), r.bh_cagr, None), (_fmt(r.cdi_cagr, "pct"), r.cdi_cagr, None),
                (f"{r.anos:.1f}", r.anos, None),
            ])
        _fill(self.t_asset, rows)

    def _fill_all(self) -> None:
        res = self.results.sort_values("excess_cdi", ascending=False, na_position="last")
        rows = []
        for r in res.itertuples():
            err = r.error if isinstance(r.error, str) else ""
            rows.append([
                (r.ticker, None, None), (r.strategy, None, None),
                (_fmt(r.excess_cdi, "pct"), r.excess_cdi, _sign_color(r.excess_cdi)),
                (_fmt(r.excess_bh, "pct"), r.excess_bh, _sign_color(r.excess_bh)),
                (_fmt(r.cagr, "pct"), r.cagr, None), (_fmt(r.sharpe, "num"), r.sharpe, None),
                (_fmt(r.max_drawdown, "pct"), r.max_drawdown, None),
                ("—" if pd.isna(r.n_trades) else f"{int(r.n_trades)}", r.n_trades, None),
                ("—" if pd.isna(r.trades_per_year) else f"{r.trades_per_year:.1f}", r.trades_per_year, None),
                (_fmt(r.exposure, "share"), r.exposure, None), (f"{r.years:.1f}", r.years, None),
                (err, None, C_SELL if err else None),
            ])
        _fill(self.t_all, rows)

    def _open_from_all(self, row: int, _col: int) -> None:
        self.open_pair.emit(self.t_all.item(row, 0).text(), self.t_all.item(row, 1).text())

    def _open_from_assets(self, row: int, _col: int) -> None:
        self.open_pair.emit(self.t_asset.item(row, 0).text(), self.t_asset.item(row, 4).text())

    def _export(self) -> None:
        if self.results is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar varredura", "varredura.csv", "CSV (*.csv)")
        if path:
            self.results.to_csv(path, index=False)
            self.log.emit(f"Varredura exportada para {path}")
