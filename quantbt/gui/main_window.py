"""Janela principal."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PySide6.QtCore import QDate, Qt, QThreadPool
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QCompleter, QDateEdit, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from quantbt.settings import app_settings
from quantbt import b3
from quantbt import external
from quantbt import data as data_mod
from quantbt.engine import BacktestConfig, BacktestResult, run_backtest
from quantbt.metrics import METRIC_INFO, OPTIMIZABLE, format_metric
from quantbt.optimizer import MAX_COMBINATIONS, count_combinations, frange, grid_search
from quantbt.strategies import BUILTIN, CUSTOM_TEMPLATE, CodeStrategy, Strategy, user_strategies_dir
from quantbt.gui.b3_tab import B3Tab
from quantbt.gui.correlation_tab import CorrelationTab
from quantbt.gui.scan_tab import ScanTab
from quantbt.gui.simulation_tab import SimulationSection
from quantbt.gui.widgets import C_BUY, C_SELL, ChartPanel, CodeEditor, NumItem, ParamForm, Worker

SRC_SAMPLE, SRC_CSV, SRC_YAHOO, SRC_B3 = "Dados de exemplo", "Arquivo CSV", "Yahoo Finance", "Base B3 (COTAHIST)"


def _readonly_table(cols: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    t.horizontalHeader().setStretchLastSection(True)
    return t


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("quantbt — Backtester de Estratégias Quantitativas")
        self.resize(1440, 900)
        self.settings = app_settings()
        self.pool = QThreadPool.globalInstance()

        self.df: pd.DataFrame | None = None
        self.result: BacktestResult | None = None
        self.strategies: list[Strategy] = list(BUILTIN)
        self.opt_worker: Worker | None = None
        self.opt_results: pd.DataFrame | None = None
        self._opt_strategy: Strategy | None = None
        self._load_seq = 0  # descarta carregamentos antigos que terminem depois de um novo
        self._run_after_load = False  # vindo da Varredura: roda o backtest quando os dados chegarem

        # seções principais: o painel lateral de backtest só aparece na seção Backtest
        self.sections = QTabWidget()
        self.sections.setDocumentMode(True)
        self.sections.setStyleSheet("QTabWidget#sections > QTabBar::tab { padding: 8px 22px; font-weight: bold; }")
        self.sections.setObjectName("sections")
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)

        self.backtest_page = QSplitter(Qt.Horizontal)
        self.backtest_page.addWidget(self._build_sidebar())
        self.backtest_page.addWidget(self._build_tabs())
        self.backtest_page.setStretchFactor(1, 1)
        self.backtest_page.setSizes([340, 1100])
        self.sections.addTab(self.backtest_page, "Backtest")

        self.corr_tab = CorrelationTab()
        self.corr_tab.log.connect(self.log)
        self.corr_tab.open_in_backtest.connect(self.open_b3_ticker)
        self.sections.addTab(self.corr_tab, "Correlação")
        self.sim_section = SimulationSection(self.log, lambda: self.risk_free.value() / 100)
        self.sections.addTab(self.sim_section, "Simulação")
        self.b3_tab = B3Tab()
        self.b3_tab.log.connect(self.log)
        self.sections.addTab(self.b3_tab, "Base B3")
        self.sections.addTab(self.log_view, "Log")
        self.setCentralWidget(self.sections)
        self._build_menu()

        self._on_strategy_changed(0)
        self._on_source_changed()
        self.load_data()

    # ================================================================ layout
    def _build_sidebar(self) -> QWidget:
        side = QWidget()
        lay = QVBoxLayout(side)

        # --- dados
        g = QGroupBox("Dados")
        f = QFormLayout(g)
        self.src = QComboBox()
        self.src.addItems([SRC_SAMPLE, SRC_B3, SRC_YAHOO, SRC_CSV])
        self.src.currentIndexChanged.connect(self._on_source_changed)
        self.ticker = QLineEdit(self.settings.value("ticker", "PETR4"))
        self.ticker.setToolTip("Código da B3 (PETR4) ou ativo externo: " + ", ".join(list(external.CATALOG)[:14]) + "…")
        completer = QCompleter(list(external.CATALOG), self.ticker)
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.ticker.setCompleter(completer)
        self.d_start = QDateEdit(QDate.currentDate().addYears(-10))
        self.d_end = QDateEdit(QDate.currentDate())
        for d in (self.d_start, self.d_end):
            d.setCalendarPopup(True)
            d.setDisplayFormat("dd/MM/yyyy")
        self.interval = QComboBox()
        self.interval.addItems(["1d", "1wk", "1mo", "1h", "30m", "15m", "5m"])
        self.csv_path = QLineEdit()
        self.csv_path.setPlaceholderText("Selecione um arquivo .csv")
        btn_browse = QPushButton("…")
        btn_browse.setFixedWidth(32)
        btn_browse.clicked.connect(self._browse_csv)
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_path)
        csv_row.addWidget(btn_browse)
        self.csv_row_w = QWidget()
        self.csv_row_w.setLayout(csv_row)
        csv_row.setContentsMargins(0, 0, 0, 0)
        self.btn_load = QPushButton("Carregar dados")
        self.btn_load.clicked.connect(self.load_data)
        self.data_info = QLabel("—")
        self.data_info.setWordWrap(True)
        self.data_info.setStyleSheet("color: #8b949e;")
        f.addRow("Fonte", self.src)
        f.addRow("Ticker", self.ticker)
        f.addRow("Início", self.d_start)
        f.addRow("Fim", self.d_end)
        f.addRow("Intervalo", self.interval)
        f.addRow("Arquivo", self.csv_row_w)
        f.addRow(self.btn_load)
        f.addRow(self.data_info)
        self.data_form = f
        lay.addWidget(g)

        # --- estratégia
        g = QGroupBox("Estratégia")
        v = QVBoxLayout(g)
        self.strat_combo = QComboBox()
        self.strat_combo.addItems([s.name for s in self.strategies])
        self.strat_combo.currentIndexChanged.connect(self._on_strategy_changed)
        self.strat_desc = QLabel()
        self.strat_desc.setWordWrap(True)
        self.strat_desc.setStyleSheet("color: #8b949e;")
        self.param_form = ParamForm()
        v.addWidget(self.strat_combo)
        v.addWidget(self.strat_desc)
        v.addWidget(self.param_form)
        lay.addWidget(g)

        # --- execução
        g = QGroupBox("Execução e custos")
        f = QFormLayout(g)
        self.capital = QDoubleSpinBox()
        self.capital.setRange(1, 1e12)
        self.capital.setDecimals(2)
        self.capital.setGroupSeparatorShown(True)
        self.capital.setValue(100_000)
        self.comm_pct = QDoubleSpinBox()
        self.comm_pct.setRange(0, 10)
        self.comm_pct.setDecimals(3)
        self.comm_pct.setSuffix(" %")
        self.comm_pct.setValue(0.05)
        self.comm_fixed = QDoubleSpinBox()
        self.comm_fixed.setRange(0, 1e6)
        self.comm_fixed.setDecimals(2)
        self.slippage = QDoubleSpinBox()
        self.slippage.setRange(0, 10)
        self.slippage.setDecimals(3)
        self.slippage.setSuffix(" %")
        self.slippage.setValue(0.05)
        self.allow_short = QCheckBox("Permitir posições vendidas")
        self.allow_short.setChecked(True)
        self.risk_free = QDoubleSpinBox()
        self.risk_free.setRange(0, 100)
        self.risk_free.setDecimals(2)
        self.risk_free.setSuffix(" % a.a.")
        self.risk_free.setValue(float(self.settings.value("risk_free", 0.0)))
        self.risk_free.setToolTip("Taxa livre de risco para Sharpe e Sortino (ex.: o CDI). Com 0, eles medem o "
                                  "retorno bruto, o que no Brasil infla muito o resultado de qualquer estratégia.")
        f.addRow("Capital inicial", self.capital)
        f.addRow("Comissão", self.comm_pct)
        f.addRow("Taxa fixa/ordem", self.comm_fixed)
        f.addRow("Slippage", self.slippage)
        f.addRow("Livre de risco", self.risk_free)
        f.addRow(self.allow_short)
        lay.addWidget(g)

        lay.addStretch(1)
        for c in (self.src, self.interval, self.strat_combo):
            c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)

        # botão fixo no rodapé: sempre visível, mesmo com muitos parâmetros
        self.btn_run = QPushButton("▶  Rodar backtest")
        self.btn_run.setMinimumHeight(38)
        self.btn_run.setStyleSheet("font-weight: bold;")
        self.btn_run.clicked.connect(self.run_backtest)
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(0, 0, 0, 6)
        pv.addWidget(scroll, 1)
        pv.addWidget(self.btn_run)
        need = side.minimumSizeHint().width() + scroll.verticalScrollBar().sizeHint().width() + 4
        panel.setMinimumWidth(max(320, need))
        return panel

    def _build_tabs(self) -> QWidget:
        self.tabs = QTabWidget()

        # gráficos
        self.charts = ChartPanel()
        self.tabs.addTab(self.charts, "Gráficos")

        # métricas
        self.metrics_table = _readonly_table(["Métrica", "Estratégia", "Buy & Hold"])
        self.tabs.addTab(self.metrics_table, "Métricas")

        # operações
        w = QWidget()
        v = QVBoxLayout(w)
        self.trades_table = _readonly_table(
            ["#", "Lado", "Entrada", "Preço entrada", "Saída", "Preço saída",
             "Quantidade", "Resultado", "Retorno", "Candles", "Status"]
        )
        self.trades_table.setSortingEnabled(True)
        btn = QPushButton("Exportar operações (CSV)…")
        btn.clicked.connect(self._export_trades)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(btn)
        v.addWidget(self.trades_table)
        v.addLayout(h)
        self.tabs.addTab(w, "Operações")

        # editor
        w = QWidget()
        v = QVBoxLayout(w)
        self.editor = CodeEditor()
        self.editor.setPlainText(self.settings.value("custom_code", CUSTOM_TEMPLATE))
        h = QHBoxLayout()
        for text, slot in (
            ("Aplicar e rodar", self._apply_custom_and_run),
            ("Aplicar", self._apply_custom),
            ("Abrir…", self._open_strategy_file),
            ("Salvar…", self._save_strategy_file),
            ("Restaurar modelo", lambda: self.editor.setPlainText(CUSTOM_TEMPLATE)),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            h.addWidget(b)
        h.addStretch(1)
        self.editor_status = QLabel()
        self.editor_status.setStyleSheet("color: #8b949e;")
        v.addLayout(h)
        v.addWidget(self.editor, 1)
        v.addWidget(self.editor_status)
        self.tabs.addTab(w, "Editor de estratégia")

        # otimização
        self.opt_tab = self._build_optimizer_tab()
        self.tabs.addTab(self.opt_tab, "Otimização")

        # varredura: várias estratégias × vários ativos
        self.scan_tab = ScanTab(self.config, lambda: list(self.strategies),
                                lambda: (self.current_strategy(), self.param_form.values()))
        self.scan_tab.log.connect(self.log)
        self.scan_tab.open_pair.connect(self.open_scan_pair)
        self.tabs.addTab(self.scan_tab, "Varredura")
        return self.tabs

    def _build_optimizer_tab(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        info = QLabel(
            "Marque os parâmetros a variar e defina a grade. Os demais usam o valor do painel lateral. "
            "Dê dois cliques em um resultado para aplicá-lo. Cuidado com sobreajuste (overfitting)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #8b949e;")
        v.addWidget(info)

        self.grid_table = QTableWidget(0, 5)
        self.grid_table.setHorizontalHeaderLabels(["Variar", "Parâmetro", "Mínimo", "Máximo", "Passo"])
        self.grid_table.verticalHeader().setVisible(False)
        self.grid_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.grid_table.setMaximumHeight(200)
        v.addWidget(self.grid_table)

        h = QHBoxLayout()
        h.addWidget(QLabel("Otimizar por:"))
        self.opt_metric = QComboBox()
        for k in OPTIMIZABLE:
            self.opt_metric.addItem(METRIC_INFO[k][0], k)
        h.addWidget(self.opt_metric)
        self.combo_count = QLabel()
        h.addWidget(self.combo_count)
        h.addStretch(1)
        self.btn_opt = QPushButton("▶  Otimizar")
        self.btn_opt.clicked.connect(self._run_optimization)
        self.btn_opt_stop = QPushButton("Parar")
        self.btn_opt_stop.setEnabled(False)
        self.btn_opt_stop.clicked.connect(self._stop_optimization)
        h.addWidget(self.btn_opt)
        h.addWidget(self.btn_opt_stop)
        v.addLayout(h)

        self.opt_progress = QProgressBar()
        self.opt_progress.setValue(0)
        v.addWidget(self.opt_progress)

        self.opt_table = _readonly_table([])
        self.opt_table.setSortingEnabled(True)
        self.opt_table.cellDoubleClicked.connect(self._apply_opt_row)
        v.addWidget(self.opt_table, 1)
        return w

    def _build_menu(self) -> None:
        m = self.menuBar().addMenu("&Arquivo")
        a = QAction("Abrir CSV…", self, shortcut=QKeySequence.Open, triggered=lambda: (self.sections.setCurrentWidget(self.backtest_page), self._menu_open_csv()))
        m.addAction(a)
        a = QAction("Exportar curva de patrimônio…", self, triggered=self._export_equity)
        m.addAction(a)
        m.addSeparator()
        m.addAction(QAction("Sair", self, shortcut=QKeySequence.Quit, triggered=self.close))
        m = self.menuBar().addMenu("&Backtest")
        m.addAction(QAction("Rodar", self, shortcut=QKeySequence("Ctrl+R"),
                            triggered=lambda: (self.run_backtest(), self.show_charts())))
        m.addAction(QAction("Otimizar", self, shortcut=QKeySequence("Ctrl+Shift+R"),
                            triggered=lambda: (self.sections.setCurrentWidget(self.backtest_page),
                                               self.tabs.setCurrentWidget(self.opt_tab),
                                               self._run_optimization())))

    # ================================================================ utilidades
    def log(self, msg: str) -> None:
        ts = pd.Timestamp.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {msg}")
        self.statusBar().showMessage(msg.splitlines()[0], 8000)

    def error(self, title: str, msg: str) -> None:
        self.log(f"ERRO — {title}: {msg}")
        QMessageBox.critical(self, title, msg.split("\n\nTraceback")[0])

    def current_strategy(self) -> Strategy:
        return self.strategies[self.strat_combo.currentIndex()]

    def config(self) -> BacktestConfig:
        return BacktestConfig(
            initial_capital=self.capital.value(),
            commission_pct=self.comm_pct.value() / 100,
            commission_fixed=self.comm_fixed.value(),
            slippage_pct=self.slippage.value() / 100,
            allow_short=self.allow_short.isChecked(),
            risk_free=self.risk_free.value() / 100,
        )

    # ================================================================ dados
    def _on_source_changed(self) -> None:
        src = self.src.currentText()
        show = {
            self.ticker: src in (SRC_YAHOO, SRC_B3), self.d_start: src in (SRC_YAHOO, SRC_B3),
            self.d_end: src in (SRC_YAHOO, SRC_B3), self.interval: src == SRC_YAHOO,
            self.csv_row_w: src == SRC_CSV,
        }
        for w, visible in show.items():
            self.data_form.setRowVisible(w, visible)

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Abrir CSV", str(Path.home()), "CSV (*.csv *.txt);;Todos (*)")
        if path:
            self.csv_path.setText(path)

    def _menu_open_csv(self) -> None:
        self.src.setCurrentText(SRC_CSV)
        self._browse_csv()
        if self.csv_path.text():
            self.load_data()

    def load_data(self) -> None:
        src = self.src.currentText()
        if src == SRC_SAMPLE:
            fn, args, label = data_mod.sample_data, (), "dados sintéticos"
        elif src == SRC_CSV:
            path = self.csv_path.text().strip()
            if not path:
                self._browse_csv()
                path = self.csv_path.text().strip()
                if not path:
                    return
            fn, args, label = data_mod.load_csv, (path,), Path(path).name
        else:
            t = self.ticker.text().strip()
            if not t:
                return
            self.settings.setValue("ticker", t)
            start = self.d_start.date().toString("yyyy-MM-dd")
            end = self.d_end.date().toString("yyyy-MM-dd")
            label = t
            if src == SRC_B3 and external.is_external(t):
                # apelidos do catálogo (DOLAR, WDO, BTC, OURO...) também funcionam nesta fonte
                fn, args = external.load, (t.upper(), start, end)
                label = f"{t.upper()} ({external.describe(t)})"
            elif src == SRC_B3:
                fn, args = b3.load_ticker, (t.upper().removesuffix(".SA"), start, end)
            else:
                fn, args = data_mod.load_yahoo, (t, start, end, self.interval.currentText())
        self.btn_load.setEnabled(False)
        self.data_info.setText("Carregando…")
        self._load_seq += 1
        seq = self._load_seq
        worker = Worker(fn, *args)
        worker.signals.finished.connect(lambda df: self._on_data_loaded(df, label, seq))
        worker.signals.error.connect(lambda msg: self._on_data_error(msg, seq))
        self.pool.start(worker)

    def _on_data_loaded(self, df: pd.DataFrame, label: str, seq: int = 0) -> None:
        if seq and seq != self._load_seq:
            return  # um carregamento mais novo já foi pedido
        self.btn_load.setEnabled(True)
        if df is None or df.empty:
            self._on_data_error("Nenhum dado no período escolhido.", seq)
            return
        self.df = df
        self.result = None
        info = (f"{label}: {len(df):,} candles\n"
                f"{df.index[0]:%d/%m/%Y} → {df.index[-1]:%d/%m/%Y}")
        self.data_info.setText(info)
        self.log(f"Dados carregados — {info.replace(chr(10), ' | ')}")
        self.charts.show_prices(df)
        if self._run_after_load:
            self._run_after_load = False
            self.run_backtest()

    def _on_data_error(self, msg: str, seq: int = 0) -> None:
        if seq and seq != self._load_seq:
            return
        self.btn_load.setEnabled(True)
        self.data_info.setText("Falha ao carregar.")
        self.error("Erro ao carregar dados", msg)

    # ================================================================ estratégia
    def _on_strategy_changed(self, i: int) -> None:
        if i < 0:
            return
        s = self.strategies[i]
        self.strat_desc.setText(s.description)
        self.param_form.set_params(s.params)
        self._refresh_grid_table()

    def _apply_custom(self) -> bool:
        code = self.editor.toPlainText()
        try:
            strat = CodeStrategy(code)
        except Exception as e:
            self.editor_status.setText(f"Erro: {e}")
            self.error("Erro na estratégia", f"{type(e).__name__}: {e}")
            return False
        self.settings.setValue("custom_code", code)
        label = f"✎ {strat.name}"
        # substitui estratégia personalizada anterior
        for i, s in enumerate(self.strategies):
            if isinstance(s, CodeStrategy):
                self.strategies[i] = strat
                self.strat_combo.setItemText(i, label)
                break
        else:
            self.strategies.append(strat)
            self.strat_combo.addItem(label)
        idx = next(i for i, s in enumerate(self.strategies) if s is strat)
        if self.strat_combo.currentIndex() == idx:
            self._on_strategy_changed(idx)
        else:
            self.strat_combo.setCurrentIndex(idx)
        self.editor_status.setText(f"Estratégia “{strat.name}” aplicada ({len(strat.params)} parâmetros).")
        self.scan_tab.refresh_strategies()
        self.log(f"Estratégia personalizada aplicada: {strat.name}")
        return True

    def _apply_custom_and_run(self) -> None:
        if self._apply_custom():
            self.run_backtest()

    def _open_strategy_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Abrir estratégia", str(user_strategies_dir()), "Python (*.py)")
        if path:
            self.editor.setPlainText(Path(path).read_text(encoding="utf-8"))
            self.editor_status.setText(f"Aberto: {path}")

    def _save_strategy_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar estratégia", str(user_strategies_dir() / "minha_estrategia.py"), "Python (*.py)"
        )
        if path:
            Path(path).write_text(self.editor.toPlainText(), encoding="utf-8")
            self.editor_status.setText(f"Salvo: {path}")

    # ================================================================ backtest
    def run_backtest(self) -> None:
        if self.df is None:
            self.error("Sem dados", "Carregue dados antes de rodar o backtest.")
            return
        strat = self.current_strategy()
        params = self.param_form.values()
        try:
            sig = strat.generate_signals(self.df, **params)
            res = run_backtest(self.df, sig, self.config(), params)
        except Exception as e:
            import traceback
            self.error("Erro no backtest", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}")
            return
        self.result = res
        self.charts.show_result(self.df, res)
        self._fill_metrics(res)
        self._fill_trades(res)
        m = res.metrics
        self.log(
            f"{strat.name} {params} → retorno {format_metric('total_return', m['total_return'])}, "
            f"Sharpe {m['sharpe']:.2f} (livre de risco {res.config.risk_free:.2%}), "
            f"DD {format_metric('max_drawdown', m['max_drawdown'])}, "
            f"{m['n_trades']} operações"
        )

    def _fill_metrics(self, res: BacktestResult) -> None:
        t = self.metrics_table
        keys = list(METRIC_INFO)
        t.setRowCount(len(keys))
        for r, k in enumerate(keys):
            t.setItem(r, 0, QTableWidgetItem(METRIC_INFO[k][0]))
            for c, src in ((1, res.metrics), (2, res.benchmark_metrics)):
                val = src.get(k)
                item = NumItem(format_metric(k, val) if val is not None else "—", val or 0)
                t.setItem(r, c, item)

    def _fill_trades(self, res: BacktestResult) -> None:
        t = self.trades_table
        t.setSortingEnabled(False)
        t.setRowCount(len(res.trades))
        green, red = QColor(C_BUY), QColor(C_SELL)
        idx = res.equity.index
        intraday = isinstance(idx, pd.DatetimeIndex) and bool((idx != idx.normalize()).any())
        dfmt = "%d/%m/%Y %H:%M" if intraday else "%d/%m/%Y"
        for r, row in enumerate(res.trades.itertuples(index=False)):
            cells = [
                NumItem(str(r + 1), r + 1),
                QTableWidgetItem(row.direction),
                QTableWidgetItem(f"{row.entry_time:{dfmt}}"),
                NumItem(f"{row.entry_price:,.2f}", row.entry_price),
                QTableWidgetItem(f"{row.exit_time:{dfmt}}"),
                NumItem(f"{row.exit_price:,.2f}", row.exit_price),
                NumItem(f"{row.quantity:,.2f}", row.quantity),
                NumItem(f"{row.pnl:,.2f}", row.pnl),
                NumItem(f"{row.return_pct * 100:,.2f}%", row.return_pct),
                NumItem(str(row.bars), row.bars),
                QTableWidgetItem(row.status),
            ]
            color = green if row.pnl > 0 else red
            cells[7].setForeground(color)
            cells[8].setForeground(color)
            for c, item in enumerate(cells):
                t.setItem(r, c, item)
        t.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        t.setSortingEnabled(True)

    def _export_trades(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar operações", "operacoes.csv", "CSV (*.csv)")
        if path:
            self.result.trades.to_csv(path, index=False)
            self.log(f"Operações exportadas para {path}")

    def _export_equity(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Exportar patrimônio", "patrimonio.csv", "CSV (*.csv)")
        if path:
            pd.concat([self.result.equity, self.result.benchmark, self.result.exposure], axis=1).to_csv(path)
            self.log(f"Curva de patrimônio exportada para {path}")

    # ================================================================ otimização
    def _refresh_grid_table(self) -> None:
        s = self.current_strategy()
        t = self.grid_table
        t.setRowCount(len(s.params))
        for r, p in enumerate(s.params):
            chk = QCheckBox()
            chk.setChecked(r < 2)
            chk.stateChanged.connect(self._update_combo_count)
            cell = QWidget()
            hl = QHBoxLayout(cell)
            hl.addWidget(chk)
            hl.setAlignment(Qt.AlignCenter)
            hl.setContentsMargins(0, 0, 0, 0)
            t.setCellWidget(r, 0, cell)
            name = QTableWidgetItem(p.label or p.name)
            name.setData(Qt.UserRole, p.name)
            name.setFlags(name.flags() & ~Qt.ItemIsEditable)
            t.setItem(r, 1, name)
            for c, val in ((2, p.min), (3, p.max), (4, p.step)):
                if p.is_int:
                    w = QSpinBox()
                    w.setRange(-10**9, 10**9)
                    w.setValue(int(val))
                else:
                    w = QDoubleSpinBox()
                    w.setDecimals(4)
                    w.setRange(-1e9, 1e9)
                    w.setSingleStep(p.step or 0.1)
                    w.setValue(float(val))
                w.valueChanged.connect(self._update_combo_count)
                t.setCellWidget(r, c, w)
        self._update_combo_count()

    def _grid(self) -> dict[str, list]:
        s = self.current_strategy()
        grid = {}
        for r, p in enumerate(s.params):
            chk = self.grid_table.cellWidget(r, 0).findChild(QCheckBox)
            if not chk.isChecked():
                continue
            lo, hi, st = (self.grid_table.cellWidget(r, c).value() for c in (2, 3, 4))
            grid[p.name] = frange(lo, hi, st, p.is_int)
        return grid

    def _update_combo_count(self) -> None:
        n = count_combinations(self._grid())
        warn = "  ⚠ acima do limite" if n > MAX_COMBINATIONS else ""
        self.combo_count.setText(f"   {n:,} combinações{warn}")

    def _run_optimization(self) -> None:
        if self.df is None:
            self.error("Sem dados", "Carregue dados antes de otimizar.")
            return
        grid = self._grid()
        if not grid:
            self.error("Otimização", "Marque ao menos um parâmetro para variar.")
            return
        n = count_combinations(grid)
        if n > MAX_COMBINATIONS:
            self.error("Otimização", f"{n:,} combinações — o limite é {MAX_COMBINATIONS:,}. Aumente os passos.")
            return
        fixed = {k: v for k, v in self.param_form.values().items() if k not in grid}
        strat = self.current_strategy()
        # lê tudo da tela AQUI (thread da interface): a thread de fundo não pode tocar nos widgets
        df, cfg = self.df, self.config()
        self._opt_strategy = strat
        self.opt_progress.setValue(0)
        self.btn_opt.setEnabled(False)
        self.btn_opt_stop.setEnabled(True)

        worker: Worker = None  # type: ignore[assignment]

        def job():
            return grid_search(
                df, strat, grid, cfg, fixed,
                progress=lambda i, total: worker.signals.progress.emit(i, total),
                should_stop=lambda: worker.cancelled,
            )

        worker = Worker(job)
        worker.signals.progress.connect(self._on_opt_progress)
        worker.signals.finished.connect(lambda df: self._on_opt_done(df, list(grid)))
        worker.signals.error.connect(self._on_opt_error)
        self.opt_worker = worker
        self.log(f"Otimizando {strat.name}: {', '.join(grid)}")
        self.pool.start(worker)

    def _stop_optimization(self) -> None:
        if self.opt_worker:
            self.opt_worker.cancelled = True

    def _on_opt_progress(self, i: int, n: int) -> None:
        self.opt_progress.setMaximum(n)
        self.opt_progress.setValue(i)

    def _on_opt_error(self, msg: str) -> None:
        self.btn_opt.setEnabled(True)
        self.btn_opt_stop.setEnabled(False)
        self.error("Erro na otimização", msg)

    def _on_opt_done(self, res: pd.DataFrame, keys: list[str]) -> None:
        self.btn_opt.setEnabled(True)
        self.btn_opt_stop.setEnabled(False)
        self.opt_results = res
        metric = self.opt_metric.currentData()
        if metric in res:
            res = res.sort_values(metric, ascending=False, na_position="last")
        cols_metrics = ["sharpe", "sortino", "cagr", "total_return", "max_drawdown", "calmar",
                        "profit_factor", "win_rate", "n_trades", "exposure"]
        cols = keys + [c for c in cols_metrics if c in res]
        t = self.opt_table
        t.setSortingEnabled(False)
        t.clear()
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels([METRIC_INFO.get(c, (c,))[0] for c in cols])
        t.setRowCount(len(res))
        for r, row in enumerate(res[cols].itertuples(index=False)):
            for c, (col, val) in enumerate(zip(cols, row)):
                text = format_metric(col, val) if col in METRIC_INFO else f"{val:g}"
                item = NumItem(text, val)
                if c == 0:
                    item.setData(Qt.UserRole, dict(zip(keys, row[: len(keys)])))
                t.setItem(r, c, item)
        if metric in cols:
            t.horizontalHeader().setSortIndicator(cols.index(metric), Qt.DescendingOrder)
        t.setSortingEnabled(True)
        best = res.iloc[0] if len(res) else None
        if best is not None and metric in res:
            self.log(f"Otimização concluída: {len(res)} combinações. Melhor {METRIC_INFO[metric][0]} = "
                     f"{format_metric(metric, best[metric])} com "
                     f"{ {k: best[k] for k in keys} }")

    def _apply_opt_row(self, row: int, _col: int) -> None:
        item = self.opt_table.item(row, 0)
        params = item.data(Qt.UserRole) if item else None
        if not params:
            return
        strat = self._opt_strategy
        if strat is not None and strat is not self.current_strategy():
            if strat not in self.strategies:
                self.error("Otimização", "A estratégia otimizada não existe mais (o código foi reaplicado).")
                return
            self.strat_combo.setCurrentIndex(self.strategies.index(strat))  # volta para ela
        self.param_form.set_values(params)
        self.run_backtest()
        self.show_charts()

    def show_charts(self) -> None:
        self.sections.setCurrentWidget(self.backtest_page)
        self.tabs.setCurrentWidget(self.charts)

    def open_scan_pair(self, ticker: str, strategy: str) -> None:
        """Abre no backtest um par da Varredura: carrega o ativo e roda a estratégia."""
        names = [s.name for s in self.strategies]
        if strategy in names:
            self.strat_combo.setCurrentIndex(names.index(strategy))
        self.src.setCurrentText(SRC_B3)
        self.ticker.setText(ticker)
        self._run_after_load = True
        self.load_data()
        self.show_charts()

    def open_b3_ticker(self, ticker: str) -> None:
        """Abre um ativo da base B3 na aba de gráficos para backtest."""
        self.src.setCurrentText(SRC_B3)
        self.ticker.setText(ticker)
        self.load_data()
        self.show_charts()

    def closeEvent(self, e) -> None:
        self.settings.setValue("custom_code", self.editor.toPlainText())
        self.settings.setValue("risk_free", self.risk_free.value())
        self._stop_optimization()
        super().closeEvent(e)
