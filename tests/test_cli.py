from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pytest

from davinci_resolve_checker.cli import app
from davinci_resolve_checker.models import CheckResult, CheckStatus
from tests.conftest import _make_state


@pytest.fixture()
def mock_probe_and_checks() -> Iterator[MagicMock]:
    state = _make_state(distro_id="arch", distro_name="Arch Linux")

    with (
        patch("davinci_resolve_checker.cli.probe_system", return_value=state),
        patch(
            "davinci_resolve_checker.cli.run_all_checks",
            return_value=[CheckResult(status=CheckStatus.PASS, message="All good")],
        ) as mock_checks,
    ):
        yield mock_checks


class TestCLI:
    def test_default_command_passes(self, mock_probe_and_checks) -> None:
        with pytest.raises(SystemExit) as exc_info:
            app([])
        assert exc_info.value.code == 0

    def test_fail_exits_nonzero(self) -> None:
        state = _make_state()
        with (
            patch("davinci_resolve_checker.cli.probe_system", return_value=state),
            patch(
                "davinci_resolve_checker.cli.run_all_checks",
                return_value=[CheckResult(status=CheckStatus.FAIL, message="Bad")],
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            app([])
        assert exc_info.value.code == 1

    def test_json_flag(self, capsys, mock_probe_and_checks) -> None:
        with pytest.raises(SystemExit):
            app(["--json"])
        captured = capsys.readouterr()
        assert '"results"' in captured.out

    def test_pro_flag_forwarded(self, mock_probe_and_checks) -> None:
        with pytest.raises(SystemExit):
            app(["--pro"])
        mock_probe_and_checks.assert_called_once()
        _, kwargs = mock_probe_and_checks.call_args
        assert kwargs.get("pro_stack") is True

    def test_fail_fast_flag_forwarded(self, mock_probe_and_checks) -> None:
        with pytest.raises(SystemExit):
            app(["--fail-fast"])
        _, kwargs = mock_probe_and_checks.call_args
        assert kwargs.get("fail_fast") is True
