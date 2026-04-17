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


class TestProbeGLInfo:
    def test_probe_gl_info(self):
        with patch("davinci_resolve_checker.probes.gpu.subprocess.run") as mock_run:
            mock_run.side_effect = [
                CompletedProcess(args="", returncode=0, stdout=" NVIDIA Corporation\n"),
                CompletedProcess(
                    args="", returncode=0, stdout="NVIDIA GeForce RTX 2070 SUPER\n"
                ),
            ]

            from davinci_resolve_checker.probes.gpu import probe_gl_info

            vendor, renderer = probe_gl_info()
            assert vendor == "NVIDIA Corporation"
            assert renderer == "NVIDIA GeForce RTX 2070 SUPER"

    def test_probe_gl_info_missing_glxinfo(self):
        with patch("davinci_resolve_checker.probes.gpu.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args="", returncode=1, stdout="")

            from davinci_resolve_checker.probes.gpu import probe_gl_info

            vendor, renderer = probe_gl_info()
            assert vendor == ""
            assert renderer == ""


class TestProbeOpenCL:
    def test_probe_opencl_platforms(self):
        import json as json_mod

        clinfo_data = {
            "platforms": [
                {
                    "CL_PLATFORM_NAME": "NVIDIA CUDA",
                    "CL_PLATFORM_ICD_SUFFIX_KHR": "NVIDIA",
                    "CL_PLATFORM_EXTENSIONS": "cl_khr_icd",
                },
            ],
            "devices": [
                {
                    "online": [
                        {"CL_DEVICE_NAME": "NVIDIA GeForce RTX 2070 SUPER"},
                    ],
                },
            ],
        }

        with patch("davinci_resolve_checker.probes.opencl.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(
                args="",
                returncode=0,
                stdout=json_mod.dumps(clinfo_data),
            )

            from davinci_resolve_checker.probes.opencl import probe_opencl_platforms

            platforms = probe_opencl_platforms()
            assert len(platforms) == 1
            assert platforms[0].name == "NVIDIA CUDA"
            assert len(platforms[0].devices) == 1

    def test_probe_opencl_platforms_failure(self):
        with patch("davinci_resolve_checker.probes.opencl.subprocess.run") as mock_run:
            mock_run.return_value = CompletedProcess(args="", returncode=1, stdout="")

            from davinci_resolve_checker.probes.opencl import probe_opencl_platforms

            assert probe_opencl_platforms() == []


class TestProbeSystem:
    def test_probe_system_integrates_all_probes(self):
        with (
            patch(
                "davinci_resolve_checker.probes.probe_chassis",
                return_value=ChassisType.DESKTOP,
            ),
            patch(
                "davinci_resolve_checker.probes.probe_distro",
                return_value=("arch", "Arch Linux"),
            ),
            patch(
                "davinci_resolve_checker.probes.probe_opencl_drivers",
                return_value=["opencl-nvidia"],
            ),
            patch(
                "davinci_resolve_checker.probes.probe_opencl_nvidia_installed",
                return_value=True,
            ),
            patch(
                "davinci_resolve_checker.probes.probe_installed_dr_package",
                return_value="davinci-resolve 19.0-1",
            ),
            patch(
                "davinci_resolve_checker.probes.probe_package_versions",
                return_value={},
            ),
            patch(
                "davinci_resolve_checker.probes.probe_roc_enable_pre_vega",
                return_value=False,
            ),
            patch("davinci_resolve_checker.probes.probe_gpus", return_value=[]),
            patch(
                "davinci_resolve_checker.probes.probe_gl_info",
                return_value=("NVIDIA Corporation", "RTX 2070"),
            ),
            patch(
                "davinci_resolve_checker.probes.probe_opencl_platforms",
                return_value=[],
            ),
        ):
            from davinci_resolve_checker.probes import probe_system

            state = probe_system()
            assert state.distro_id == "arch"
            assert state.chassis == ChassisType.DESKTOP
            assert state.gl_vendor == "NVIDIA Corporation"
