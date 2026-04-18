from __future__ import annotations

import json

from davinci_resolve_checker.models import CheckResult, CheckStatus
from davinci_resolve_checker.render import render_json, render_text
from tests.conftest import _make_state


class TestRenderJson:
    def test_valid_json_output(self, capsys, nvidia_desktop):
        results = [
            CheckResult(status=CheckStatus.PASS, message="All good"),
        ]
        render_json(nvidia_desktop, results)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is True
        assert "system" in data
        assert "results" in data
        assert data["results"][0]["status"] == "pass"

    def test_system_state_serialized(self, capsys, nvidia_desktop):
        render_json(nvidia_desktop, [])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["system"]["distro_id"] == "arch"

    def test_failed_json_output_sets_ok_false(self, capsys, nvidia_desktop):
        render_json(nvidia_desktop, [CheckResult(status=CheckStatus.FAIL, message="Bad")])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["ok"] is False


class TestRenderText:
    def test_renders_without_error(self, capsys, nvidia_desktop):
        results = [
            CheckResult(status=CheckStatus.PASS, message="All good"),
            CheckResult(status=CheckStatus.FAIL, message="Bad thing", suggestion="Fix it"),
        ]
        render_text(nvidia_desktop, results)
        captured = capsys.readouterr()
        assert "All good" in captured.out
        assert "Bad thing" in captured.out
        assert "Fix it" in captured.out
        assert captured.out.index("Bad thing") < captured.out.index("All good")

    def test_shows_gpu_info(self, capsys, nvidia_desktop):
        render_text(nvidia_desktop, [])
        captured = capsys.readouterr()
        assert "RTX 2070" in captured.out

    def test_shows_distro(self, capsys, nvidia_desktop):
        render_text(nvidia_desktop, [])
        captured = capsys.readouterr()
        assert "Arch Linux" in captured.out

    def test_empty_results(self, capsys):
        state = _make_state()
        render_text(state, [])
        captured = capsys.readouterr()
        assert len(captured.out) > 0
