"""Componentes reutilizáveis da interface."""

from __future__ import annotations

import traceback

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, QRegularExpression, QRunnable, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QSyntaxHighlighter, QTextCharFormat
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFormLayout, QPlainTextEdit, QSpinBox, QTableWidgetItem, QWidget,
)

from quantbt.engine import BacktestResult
from quantbt.metrics import drawdown
from quantbt.strategies import Param

pg.setConfigOptions(antialias=True, background="#15181e", foreground="#c8ccd4")

C_STRAT = "#4f9dff"
C_BENCH = "#9aa0a6"
C_PRICE = "#d7dae0"
C_BUY = "#2ecc71"
C_SELL = "#ff5c5c"
C_DD = "#ff7b72"


# ---------------------------------------------------------------- tabelas
class NumItem(QTableWidgetItem):
    """Célula de tabela alinhada à direita que ordena pelo valor numérico (vazios vão para o fim)."""

    def __init__(self, text: str, value) -> None:
        super().__init__(text)
        try:
            v = float(value)
        except (TypeError, ValueError):
            v = float("nan")
        self.value = -np.inf if np.isnan(v) else v
        self.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

    def __lt__(self, other) -> bool:
        if isinstance(other, NumItem):
            return self.value < other.value
        return super().__lt__(other)


# ---------------------------------------------------------------- tarefas em segundo plano
class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(int, int)


class Worker(QRunnable):
    """Executa uma função fora da thread da interface."""

    # mantém tarefas vivas até terminarem (senão o Python pode coletá-las antes de emitir o sinal)
    _active: set["Worker"] = set()

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self.setAutoDelete(False)
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()
        self.cancelled = False
        Worker._active.add(self)
        self.signals.finished.connect(self._release)
        self.signals.error.connect(self._release)

    def _release(self, *_):
        # adia a liberação para depois dos outros slots conectados
        QTimer.singleShot(0, lambda: Worker._active.discard(self))

    @Slot()
    def run(self) -> None:
        try:
            out = self.fn(*self.args, **self.kwargs)
        except Exception as e:
            self.signals.error.emit(f"{e}\n\n{traceback.format_exc()}")
        else:
            self.signals.finished.emit(out)


# ---------------------------------------------------------------- formulário de parâmetros
class ParamForm(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout_ = QFormLayout(self)
        self.layout_.setContentsMargins(0, 0, 0, 0)
        self.editors: dict[str, QSpinBox | QDoubleSpinBox] = {}

    def set_params(self, params: list[Param]) -> None:
        while self.layout_.rowCount():
            self.layout_.removeRow(0)
        self.editors.clear()
        for p in params:
            if p.is_int:
                w = QSpinBox()
                w.setRange(int(min(p.min, p.default)) - 10**6, int(max(p.max, p.default)) + 10**6)
                w.setSingleStep(int(p.step) or 1)
                w.setValue(int(p.default))
            else:
                w = QDoubleSpinBox()
                w.setDecimals(4)
                w.setRange(-1e9, 1e9)
                w.setSingleStep(p.step or 0.1)
                w.setValue(float(p.default))
            self.editors[p.name] = w
            self.layout_.addRow(p.label or p.name, w)

    def values(self) -> dict:
        return {k: w.value() for k, w in self.editors.items()}

    def set_values(self, values: dict) -> None:
        for k, v in values.items():
            if k in self.editors and v is not None and not (isinstance(v, float) and np.isnan(v)):
                w = self.editors[k]
                w.setValue(int(v) if isinstance(w, QSpinBox) else float(v))


# ---------------------------------------------------------------- gráficos
def _ts(index) -> np.ndarray:
    """Datas -> segundos desde 1970 (independe da resolução: ns, ms ou s)."""
    return pd.DatetimeIndex(index).values.astype("datetime64[ms]").astype(np.int64) / 1e3


class ChartPanel(pg.GraphicsLayoutWidget):
    """Preço com operações, curva de patrimônio e drawdown com eixo X sincronizado."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.price = self.addPlot(row=0, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self.equity = self.addPlot(row=1, col=0, axisItems={"bottom": pg.DateAxisItem()})
        self.dd = self.addPlot(row=2, col=0, axisItems={"bottom": pg.DateAxisItem()})
        for p, title in ((self.price, "Preço e operações"), (self.equity, "Patrimônio"), (self.dd, "Drawdown")):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setTitle(title, size="10pt")
            p.getAxis("left").setWidth(70)
        self.equity.setXLink(self.price)
        self.dd.setXLink(self.price)
        self.ci.layout.setRowStretchFactor(0, 5)
        self.ci.layout.setRowStretchFactor(1, 3)
        self.ci.layout.setRowStretchFactor(2, 2)
        self.equity.addLegend(offset=(10, 10))
        self.price.addLegend(offset=(10, 10))

    def clear_all(self) -> None:
        for p in (self.price, self.equity, self.dd):
            p.clear()

    def show_prices(self, df: pd.DataFrame) -> None:
        self.clear_all()
        self.price.plot(_ts(df.index), df["close"].to_numpy(), pen=pg.mkPen(C_PRICE, width=1), name="Fechamento")
        self.price.autoRange()

    def show_result(self, df: pd.DataFrame, res: BacktestResult) -> None:
        self.clear_all()
        x = _ts(df.index)
        self.price.plot(x, df["close"].to_numpy(), pen=pg.mkPen(C_PRICE, width=1), name="Fechamento")

        fills = res.fills
        if len(fills):
            fx = _ts(pd.DatetimeIndex(fills["time"]))
            buys = fills["quantity"].to_numpy() > 0
            py = fills["price"].to_numpy()
            self.price.plot(fx[buys], py[buys], pen=None, symbol="t1", symbolSize=10,
                            symbolBrush=C_BUY, symbolPen=None, name="Compra")
            self.price.plot(fx[~buys], py[~buys], pen=None, symbol="t", symbolSize=10,
                            symbolBrush=C_SELL, symbolPen=None, name="Venda")

        self.equity.plot(x, res.benchmark.to_numpy(), pen=pg.mkPen(C_BENCH, width=1, style=Qt.DashLine), name="Buy & Hold")
        self.equity.plot(x, res.equity.to_numpy(), pen=pg.mkPen(C_STRAT, width=2), name="Estratégia")

        dd = drawdown(res.equity).to_numpy() * 100
        self.dd.plot(x, dd, pen=pg.mkPen(C_DD, width=1), fillLevel=0, brush=pg.mkBrush(255, 123, 114, 70))
        self.dd.getAxis("left").setLabel("%")
        self.price.autoRange()
        self.equity.autoRange()
        self.dd.autoRange()


# ---------------------------------------------------------------- editor de código
class PythonHighlighter(QSyntaxHighlighter):
    KEYWORDS = (
        "and as assert break class continue def del elif else except False finally for from "
        "global if import in is lambda None nonlocal not or pass raise return True try while with yield"
    ).split()

    def __init__(self, doc) -> None:
        super().__init__(doc)

        def fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
            f = QTextCharFormat()
            f.setForeground(QColor(color))
            if bold:
                f.setFontWeight(QFont.Bold)
            f.setFontItalic(italic)
            return f

        self.rules = [
            (QRegularExpression(r"\b(" + "|".join(self.KEYWORDS) + r")\b"), fmt("#c678dd", bold=True)),
            (QRegularExpression(r"\b(pd|np|sma|ema|rsi|atr|df)\b"), fmt("#61afef")),
            (QRegularExpression(r"\b[0-9]+(\.[0-9]+)?\b"), fmt("#d19a66")),
            (QRegularExpression(r"\bdef\s+(\w+)"), fmt("#e5c07b")),
            (QRegularExpression(r"\"[^\"\\]*(\\.[^\"\\]*)*\"|'[^'\\]*(\\.[^'\\]*)*'"), fmt("#98c379")),
            (QRegularExpression(r"#[^\n]*"), fmt("#7f848e", italic=True)),
        ]

    def highlightBlock(self, text: str) -> None:
        for rx, f in self.rules:
            it = rx.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), f)


class CodeEditor(QPlainTextEdit):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSize(12)
        self.setFont(font)
        self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(" "))
        self.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.highlighter = PythonHighlighter(self.document())

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Tab:
            self.insertPlainText("    ")
            return
        super().keyPressEvent(e)
