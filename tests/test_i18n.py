from __future__ import annotations

import builtins

from davinci_resolve_checker.i18n import setup_i18n


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
