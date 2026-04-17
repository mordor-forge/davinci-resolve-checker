from __future__ import annotations

import builtins
from unittest.mock import patch

from davinci_resolve_checker.i18n import _detect_locale, setup_i18n


class TestI18n:
    def test_setup_default_locale(self):
        trans = setup_i18n()
        assert trans is not None

    def test_setup_explicit_locale(self):
        trans = setup_i18n("en_US")
        assert trans is not None

    def test_fallback_to_english(self):
        trans = setup_i18n("xx_XX")
        assert trans is not None

    def test_gettext_function_works(self):
        setup_i18n("en_US")
        assert hasattr(builtins, "_")


class TestDetectLocale:
    def test_detect_locale_returns_system_locale(self):
        with patch(
            "davinci_resolve_checker.i18n.locale.getlocale", return_value=("it_IT", "UTF-8")
        ):
            assert _detect_locale() == "it_IT"

    def test_detect_locale_none_returns_default(self):
        with patch("davinci_resolve_checker.i18n.locale.getlocale", return_value=(None, None)):
            assert _detect_locale() == "en_US"

    def test_detect_locale_exception_returns_default(self):
        with patch("davinci_resolve_checker.i18n.locale.getlocale", side_effect=ValueError("boom")):
            assert _detect_locale() == "en_US"
