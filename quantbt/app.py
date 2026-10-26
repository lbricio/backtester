"""Ponto de entrada da aplicação."""

from __future__ import annotations

import sys


def _dark_palette(app) -> None:
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")  # visual idêntico em Linux e macOS
    p = QPalette()
    bg, base, text = QColor("#1e2229"), QColor("#15181e"), QColor("#d7dae0")
    p.setColor(QPalette.Window, bg)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, QColor("#1b1f26"))
    p.setColor(QPalette.ToolTipBase, bg)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, QColor("#2a2f38"))
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.Highlight, QColor("#3d7bd9"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.PlaceholderText, QColor("#6b7280"))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        p.setColor(QPalette.Disabled, role, QColor("#6b7280"))
    app.setPalette(p)


def _install_crash_logging():
    """Grava erros não tratados em ~/.quantbt/erros.log e falhas graves (travamentos do Python/Qt)
    em ~/.quantbt/falhas.log, para dar para descobrir a causa depois."""
    import datetime
    import faulthandler
    import traceback
    from pathlib import Path

    base = Path.home() / ".quantbt"
    base.mkdir(parents=True, exist_ok=True)
    fault_file = open(base / "falhas.log", "a", buffering=1)  # mantido aberto durante a execução
    fault_file.write(f"\n=== início {datetime.datetime.now():%d/%m/%Y %H:%M:%S} ===\n")
    faulthandler.enable(fault_file, all_threads=True)

    def hook(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        with open(base / "erros.log", "a") as f:
            f.write(f"\n=== {datetime.datetime.now():%d/%m/%Y %H:%M:%S} ===\n{text}")
        sys.__stderr__.write(text)
        try:  # avisa na tela em vez de fechar o app
            from PySide6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance():
                QMessageBox.critical(None, "Erro inesperado",
                                     f"{exc_type.__name__}: {exc}\n\nDetalhes gravados em {base / 'erros.log'}")
        except Exception:  # noqa: BLE001
            pass

    sys.excepthook = hook
    return fault_file


def main() -> None:
    from PySide6.QtWidgets import QApplication

    from quantbt.gui.main_window import MainWindow

    _fault_file = _install_crash_logging()  # noqa: F841 — precisa ficar aberto
    app = QApplication(sys.argv)
    app.setApplicationName("quantbt")
    app.setOrganizationName("quantbt")
    _dark_palette(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
