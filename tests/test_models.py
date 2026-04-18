from __future__ import annotations

import pytest

from davinci_resolve_checker.models import (
    ChassisType,
    CheckResult,
    CheckStatus,
    GPUDevice,
    GPUVendor,
    OpenCLPlatform,
    SystemState,
)


class TestGPUVendor:
    def test_vendor_values(self):
        assert GPUVendor.INTEL == "Intel Corporation"
        assert GPUVendor.AMD == "Advanced Micro Devices, Inc. [AMD/ATI]"
        assert GPUVendor.NVIDIA == "NVIDIA Corporation"


class TestChassisType:
    @pytest.mark.parametrize(
        "chassis",
        [
            ChassisType.LAPTOP,
            ChassisType.NOTEBOOK,
            ChassisType.CONVERTIBLE,
            ChassisType.PORTABLE,
            ChassisType.TABLET,
            ChassisType.DETACHABLE,
            ChassisType.HAND_HELD,
            ChassisType.SUB_NOTEBOOK,
        ],
    )
    def test_mobile_types(self, chassis: ChassisType):
        assert chassis.is_mobile is True
        assert chassis.is_desktop is False

    @pytest.mark.parametrize(
        "chassis",
        [
            ChassisType.DESKTOP,
            ChassisType.ALL_IN_ONE,
            ChassisType.MINI_PC,
            ChassisType.SPACE_SAVING,
            ChassisType.TOWER,
            ChassisType.OTHER,
        ],
    )
    def test_desktop_types(self, chassis: ChassisType):
        assert chassis.is_desktop is True
        assert chassis.is_mobile is False

    def test_unsupported_chassis(self):
        assert ChassisType.BLADE.is_mobile is False
        assert ChassisType.BLADE.is_desktop is False

    def test_chassis_from_string(self):
        assert ChassisType("Laptop") == ChassisType.LAPTOP
        assert ChassisType("Mini PC") == ChassisType.MINI_PC


class TestGPUDevice:
    def test_is_pre_vega_ellesmere(self):
        gpu = GPUDevice(
            name="Ellesmere [Radeon RX 470/480/570/580]",
            vendor=GPUVendor.AMD,
            driver="amdgpu",
            kernel_modules=["amdgpu"],
            pci_slot="0000:01:00.0",
            pci_class=0x0300,
        )
        assert gpu.is_pre_vega is True

    def test_is_pre_vega_navi(self):
        gpu = GPUDevice(
            name="Navi 23 [Radeon RX 6600]",
            vendor=GPUVendor.AMD,
            driver="amdgpu",
            kernel_modules=["amdgpu"],
            pci_slot="0000:01:00.0",
            pci_class=0x0300,
        )
        assert gpu.is_pre_vega is False

    @pytest.mark.parametrize("codename", ["Vega", "Cezanne", "Raphael", "Barcelo", "Rembrandt"])
    def test_is_pre_vega_modern_codenames(self, codename: str):
        gpu = GPUDevice(
            name=f"{codename} [Some GPU]",
            vendor=GPUVendor.AMD,
            driver="amdgpu",
            kernel_modules=["amdgpu"],
            pci_slot="0000:01:00.0",
            pci_class=0x0300,
        )
        assert gpu.is_pre_vega is False

    def test_is_pre_vega_unknown_amd(self):
        gpu = GPUDevice(
            name="Some Unknown AMD GPU",
            vendor=GPUVendor.AMD,
            driver="amdgpu",
            kernel_modules=["amdgpu"],
            pci_slot="0000:01:00.0",
            pci_class=0x0300,
        )
        assert gpu.is_pre_vega is None

    def test_is_pre_vega_nvidia_returns_none(self):
        gpu = GPUDevice(
            name="RTX 2070 SUPER",
            vendor=GPUVendor.NVIDIA,
            driver="nvidia",
            kernel_modules=["nvidia"],
            pci_slot="0000:01:00.0",
            pci_class=0x0300,
        )
        assert gpu.is_pre_vega is None

    def test_is_pre_vega_intel_returns_none(self):
        gpu = GPUDevice(
            name="UHD Graphics 630",
            vendor=GPUVendor.INTEL,
            driver="i915",
            kernel_modules=["i915"],
            pci_slot="0000:00:02.0",
            pci_class=0x0300,
        )
        assert gpu.is_pre_vega is None


class TestOpenCLPlatform:
    def test_is_orca(self):
        platform = OpenCLPlatform(
            name="AMD Accelerated Parallel Processing",
            icd_suffix="AMD",
            extensions="cl_amd_offline_devices cl_khr_icd",
            devices=[],
        )
        assert platform.is_orca is True
        assert platform.is_roc is False

    def test_is_roc(self):
        platform = OpenCLPlatform(
            name="AMD Accelerated Parallel Processing",
            icd_suffix="AMD",
            extensions="cl_khr_icd cl_khr_byte_addressable_store",
            devices=[],
        )
        assert platform.is_roc is True
        assert platform.is_orca is False

    def test_non_amd_platform(self):
        platform = OpenCLPlatform(
            name="NVIDIA CUDA",
            icd_suffix="NV",
            extensions="cl_khr_icd",
            devices=[],
        )
        assert platform.is_orca is False
        assert platform.is_roc is False
        assert platform.is_nvidia is True
        assert platform.is_amd is False
        assert platform.has_devices is False

    def test_clover_platform(self):
        platform = OpenCLPlatform(
            name="Clover",
            icd_suffix="MESA",
            extensions="cl_khr_icd",
            devices=[],
        )
        assert platform.is_clover is True


class TestSystemState:
    def test_serialization_roundtrip(self):
        state = SystemState(
            distro_id="arch",
            distro_name="Arch Linux",
            chassis=ChassisType.DESKTOP,
            gpus=[
                GPUDevice(
                    name="RTX 2070",
                    vendor=GPUVendor.NVIDIA,
                    driver="nvidia",
                    kernel_modules=["nvidia"],
                    pci_slot="0000:01:00.0",
                    pci_class=0x0300,
                )
            ],
            opencl_drivers=["opencl-nvidia"],
            opencl_platforms=[],
            opencl_nvidia_installed=True,
            gl_vendor="NVIDIA Corporation",
            gl_renderer="NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2",
            installed_dr_package="davinci-resolve 19.0-1",
        )
        json_str = state.model_dump_json()
        restored = SystemState.model_validate_json(json_str)
        assert restored == state

    def test_defaults(self):
        state = SystemState(
            distro_id="arch",
            distro_name="Arch Linux",
            chassis=ChassisType.DESKTOP,
            gpus=[],
            opencl_drivers=[],
            opencl_platforms=[],
            opencl_nvidia_installed=False,
            gl_vendor="",
            gl_renderer="",
            installed_dr_package=None,
        )
        assert state.package_versions == {}
        assert state.roc_enable_pre_vega is False


class TestCheckResult:
    def test_pass_result(self):
        result = CheckResult(status=CheckStatus.PASS, message="All good")
        assert result.suggestion is None

    def test_fail_with_suggestion(self):
        result = CheckResult(
            status=CheckStatus.FAIL,
            message="Missing driver",
            suggestion="Install opencl-nvidia",
        )
        assert result.status == CheckStatus.FAIL
        assert result.suggestion == "Install opencl-nvidia"
