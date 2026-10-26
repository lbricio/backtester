"""Preferências salvas do app (listas de ativos, último ticker, código do editor...).

Por padrão ficam no local do sistema (no macOS, ~/Library/Preferences/com.quantbt.quantbt.plist).
Com a variável de ambiente QUANTBT_CONFIG_DIR, ficam num arquivo .ini nessa pasta — usado pelos
testes automáticos para nunca mexer nas preferências reais do usuário.
"""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QSettings


def app_settings() -> QSettings:
    custom = os.environ.get("QUANTBT_CONFIG_DIR")
    if custom:
        Path(custom).mkdir(parents=True, exist_ok=True)
        return QSettings(str(Path(custom) / "quantbt.ini"), QSettings.IniFormat)
    return QSettings("quantbt", "quantbt")
