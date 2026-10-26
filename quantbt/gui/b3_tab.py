"""Aba de gerenciamento da base histórica da B3."""

from __future__ import annotations

from datetime import date

from PySide6.QtCore import QThreadPool, Signal
from PySide6.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from quantbt import b3
from quantbt import external
from quantbt.gui.widgets import Worker


class B3Tab(QWidget):
    log = Signal(str)
    _line = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.pool = QThreadPool.globalInstance()
        self.worker: Worker | None = None
        v = QVBoxLayout(self)

        info = QLabel(
            "Séries históricas oficiais da B3 (arquivos COTAHIST anuais, mercado à vista: ações, units, "
            "FIIs, Fiagro, ETFs e BDRs; direitos de subscrição são descartados). Os arquivos são guardados em "
            f"<code>{b3.store_dir()}</code>.<br>"
            "<b>Importante:</b> os preços da B3 não são ajustados por proventos. O app corrige "
            "desdobramentos/grupamentos detectáveis, mas dividendos não são descontados."
        )
        info.setWordWrap(True)
        v.addWidget(info)

        g = QGroupBox("Baixar da B3")
        h = QHBoxLayout(g)
        this_year = date.today().year
        self.y0 = QSpinBox()
        self.y0.setRange(b3.FIRST_YEAR, this_year)
        self.y0.setValue(this_year - 15)
        self.y1 = QSpinBox()
        self.y1.setRange(b3.FIRST_YEAR, this_year)
        self.y1.setValue(this_year)
        self.btn_dl = QPushButton("Baixar anos")
        self.btn_dl.clicked.connect(self._download)
        self.btn_missing = QPushButton("Baixar só os que faltam")
        self.btn_missing.clicked.connect(lambda: self._download(only_missing=True))
        self.btn_update = QPushButton(f"Atualizar {this_year}")
        self.btn_update.clicked.connect(lambda: self._start([this_year]))
        self.btn_stop = QPushButton("Parar")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop)
        h.addWidget(QLabel("De"))
        h.addWidget(self.y0)
        h.addWidget(QLabel("até"))
        h.addWidget(self.y1)
        for b in (self.btn_dl, self.btn_missing, self.btn_update, self.btn_stop):
            h.addWidget(b)
        h.addStretch(1)
        v.addWidget(g)

        g = QGroupBox("Importar arquivos baixados manualmente")
        h = QHBoxLayout(g)
        btn = QPushButton("Importar COTAHIST (.ZIP ou .TXT)…")
        btn.clicked.connect(self._import)
        h.addWidget(btn)
        h.addWidget(QLabel("Útil se o download direto falhar (rede corporativa, proxy etc.)."))
        h.addStretch(1)
        v.addWidget(g)

        g = QGroupBox("Dados externos: dólar, mini dólar, commodities, cripto e índices")
        ev = QVBoxLayout(g)
        cat = external.catalog_table()
        lines = []
        for grp, sub in cat.groupby("grupo", sort=False):
            lines.append(f"<b>{grp}:</b> " + ", ".join(f"{r.apelido} <span style='color:#8b949e'>({r.nome})</span>"
                                                      for r in sub.itertuples()))
        lbl = QLabel("<br>".join(lines) + "<br><span style='color:#8b949e'>Use os apelidos na Correlação e no "
                     "Backtest (fonte Base B3). Futuros WDO/DOL/WIN/IND não estão no COTAHIST: usam o dólar à "
                     "vista e o Ibovespa como substitutos. Fonte: Yahoo Finance e Banco Central (PTAX).</span>")
        lbl.setWordWrap(True)
        ev.addWidget(lbl)
        h2 = QHBoxLayout()
        self.btn_ext = QPushButton("Atualizar dados externos (10 anos)")
        self.btn_ext.setToolTip(f"Baixa/atualiza todos os ativos do catálogo em {external.cache_dir()}")
        self.btn_ext.clicked.connect(self._update_external)
        h2.addWidget(self.btn_ext)
        h2.addWidget(QLabel("Atualizam sozinhos quando usados, se o cache tiver mais de 12 h."))
        h2.addStretch(1)
        ev.addLayout(h2)
        v.addWidget(g)

        self.years_label = QLabel()
        self.years_label.setWordWrap(True)
        v.addWidget(self.years_label)

        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        v.addWidget(self.out, 1)
        self._line.connect(self._append)
        self.refresh()

    def _append(self, msg: str) -> None:
        self.out.appendPlainText(msg)
        self.log.emit(f"B3: {msg}")

    def refresh(self) -> None:
        ys = b3.available_years()
        size = sum(b3.year_path(y).stat().st_size for y in ys) / 1e6
        self.years_label.setText(
            f"<b>Anos na base:</b> {self._ranges(ys) if ys else 'nenhum'}"
            + (f" &nbsp;({size:,.0f} MB)" if ys else "")
        )

    @staticmethod
    def _ranges(ys: list[int]) -> str:
        out, start, prev = [], ys[0], ys[0]
        for y in ys[1:] + [None]:
            if y is not None and y == prev + 1:
                prev = y
                continue
            out.append(str(start) if start == prev else f"{start}–{prev}")
            if y is not None:
                start = prev = y
        return ", ".join(out)

    def _download(self, only_missing: bool = False) -> None:
        y0, y1 = sorted((self.y0.value(), self.y1.value()))
        years = list(range(y0, y1 + 1))
        if only_missing:
            have = set(b3.available_years())
            years = [y for y in years if y not in have]
            if not years:
                self._append("Todos os anos do intervalo já estão na base.")
                return
        self._start(years)

    def _start(self, years: list[int]) -> None:
        self._set_busy(True)
        worker: Worker = None  # type: ignore[assignment]

        def job():
            return b3.download_years(years, log=self._line.emit, should_stop=lambda: worker.cancelled)

        worker = Worker(job)
        worker.signals.finished.connect(self._done)
        worker.signals.error.connect(lambda m: (self._append(f"Erro: {m.splitlines()[0]}"), self._set_busy(False)))
        self.worker = worker
        self.pool.start(worker)

    def _done(self, years: list[int]) -> None:
        self._set_busy(False)
        self._append(f"Concluído: {len(years)} ano(s) atualizados.")
        self.refresh()

    def _stop(self) -> None:
        if self.worker:
            self.worker.cancelled = True
            self._append("Parando após o ano atual…")

    def _set_busy(self, busy: bool) -> None:
        for b in (self.btn_dl, self.btn_missing, self.btn_update):
            b.setEnabled(not busy)
        self.btn_stop.setEnabled(busy)

    def _update_external(self) -> None:
        self.btn_ext.setEnabled(False)
        self._append("Atualizando dados externos…")
        w = Worker(external.update_all, None, None, self._line.emit)
        w.signals.finished.connect(lambda done: (self.btn_ext.setEnabled(True),
                                                 self._append(f"Dados externos: {len(done)} séries atualizadas.")))
        w.signals.error.connect(lambda m: (self.btn_ext.setEnabled(True), self._append(f"Erro: {m.splitlines()[0]}")))
        self.pool.start(w)

    def _import(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Importar COTAHIST", "", "COTAHIST (*.zip *.ZIP *.txt *.TXT)")
        if not paths:
            return
        self._set_busy(True)

        def job():
            out = []
            for p in paths:
                try:
                    y, n = b3.import_file(p)
                    self._line.emit(f"Importado {p}: ano {y}, {n:,} registros.")
                    out.append(y)
                except Exception as e:
                    self._line.emit(f"Falha ao importar {p}: {e}")
            return out

        w = Worker(job)
        w.signals.finished.connect(self._done)
        self.pool.start(w)
