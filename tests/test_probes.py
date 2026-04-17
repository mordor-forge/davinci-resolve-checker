from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import mock_open, patch

from davinci_resolve_checker.models import ChassisType


class TestProbeChassis:
    @patch("builtins.open", mock_open(read_data="3\n"))
    def test_probe_chassis_desktop(self):
        from davinci_resolve_checker.probes.system import probe_chassis

        assert probe_chassis() == ChassisType.DESKTOP

    @patch("builtins.open", mock_open(read_data="9\n"))
    def test_probe_chassis_laptop(self):
        from davinci_resolve_checker.probes.system import probe_chassis

        assert probe_chassis() == ChassisType.LAPTOP

    @patch("builtins.open", mock_open(read_data="35\n"))
    def test_probe_chassis_mini_pc(self):
        from davinci_resolve_checker.probes.system import probe_chassis

        assert probe_chassis() == ChassisType.MINI_PC


class TestProbeDistro:
    def test_probe_distro(self):
        with patch("davinci_resolve_checker.probes.system.distro") as mock_distro:
            mock_distro.id.return_value = "arch"
            mock_distro.name.return_value = "Arch Linux"

            from davinci_resolve_checker.probes.system import probe_distro

            distro_id, distro_name = probe_distro()
            assert distro_id == "arch"
            assert distro_name == "Arch Linux"


class TestProbeOpenCLDrivers:
    def test_probe_opencl_drivers(self):
        with patch("davinci_resolve_checker.probes.system.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args="",
                returncode=0,
                stdout="opencl-nvidia 560.35.03-1\nrocm-opencl-runtime 6.1.0-1\n",
            )

            from davinci_resolve_checker.probes.system import probe_opencl_drivers

            drivers = probe_opencl_drivers()
            assert drivers == ["opencl-nvidia 560.35.03-1", "rocm-opencl-runtime 6.1.0-1"]

    def test_probe_opencl_drivers_none_found(self):
        with patch("davinci_resolve_checker.probes.system.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args="", returncode=1, stdout="")

            from davinci_resolve_checker.probes.system import probe_opencl_drivers

            assert probe_opencl_drivers() == []


class TestProbeRocEnablePreVega:
    def test_probe_roc_enable_pre_vega_set(self):
        from davinci_resolve_checker.probes.system import probe_roc_enable_pre_vega

        with patch.dict("os.environ", {"ROC_ENABLE_PRE_VEGA": "1"}):
            assert probe_roc_enable_pre_vega() is True

    def test_probe_roc_enable_pre_vega_unset(self):
        from davinci_resolve_checker.probes.system import probe_roc_enable_pre_vega

        with patch.dict("os.environ", {}, clear=True):
            assert probe_roc_enable_pre_vega() is False
