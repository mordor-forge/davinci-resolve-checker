from __future__ import annotations

import gettext
import locale
from importlib import resources


def setup_i18n(
    preferred_locale: str | None = None,
) -> gettext.GNUTranslations | gettext.NullTranslations:
    loc = preferred_locale or _detect_locale()
    locale_dir = str(resources.files("davinci_resolve_checker") / "locale")

    try:
        trans = gettext.translation("messages", localedir=locale_dir, languages=[loc])
    except FileNotFoundError:
        try:
            trans = gettext.translation("messages", localedir=locale_dir, languages=["en_US"])
        except FileNotFoundError:
            trans = gettext.NullTranslations()

    trans.install()
    return trans


def _detect_locale() -> str:
    try:
        loc, _ = locale.getlocale()
        return loc or "en_US"
    except Exception:
        return "en_US"
