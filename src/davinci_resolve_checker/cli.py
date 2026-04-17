from __future__ import annotations

import sys

import cyclopts

from davinci_resolve_checker.checks import run_all_checks
from davinci_resolve_checker.i18n import setup_i18n
from davinci_resolve_checker.models import CheckStatus
from davinci_resolve_checker.probes import probe_system
from davinci_resolve_checker.render import render_json, render_text

app = cyclopts.App(
    name="davinci-resolve-checker",
    help="Check system configuration for DaVinci Resolve compatibility on Arch-based Linux.",
)


@app.default
def check(
    *,
    locale: str | None = None,
    pro: bool = False,
    fail_fast: bool = False,
    json: bool = False,
) -> None:
    """Run all compatibility checks and report results."""
    setup_i18n(locale)
    state = probe_system()
    results = run_all_checks(state, pro_stack=pro, fail_fast=fail_fast)

    if json:
        render_json(state, results)
    else:
        render_text(state, results)

    sys.exit(1 if any(r.status == CheckStatus.FAIL for r in results) else 0)
