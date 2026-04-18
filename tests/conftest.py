from __future__ import annotations

import pytest

from davinci_resolve_checker.models import (
    ChassisType,
    GPUDevice,
    GPUVendor,
    OpenCLDevice,
    OpenCLPlatform,
    SystemState,
)


def _make_gpu(
    name: str = "RTX 2070 SUPER",
    vendor: GPUVendor = GPUVendor.NVIDIA,
    driver: str | None = "nvidia",
    kernel_modules: list[str] | None = None,
    pci_slot: str = "0000:01:00.0",
    pci_class: int = 0x0300,
) -> GPUDevice:
    return GPUDevice(
        name=name,
        vendor=vendor,
        driver=driver,
        kernel_modules=kernel_modules or [driver or ""],
        pci_slot=pci_slot,
        pci_class=pci_class,
    )


def _make_state(**overrides) -> SystemState:
    defaults = {
        "distro_id": "arch",
        "distro_name": "Arch Linux",
        "chassis": ChassisType.DESKTOP,
        "gpus": [],
        "opencl_drivers": [],
        "opencl_platforms": [],
        "opencl_nvidia_installed": False,
        "gl_vendor": "",
        "gl_renderer": "",
        "installed_dr_package": None,
        "package_versions": {},
        "roc_enable_pre_vega": False,
    }
    defaults.update(overrides)
    return SystemState(**defaults)


INTEL_GPU = _make_gpu(
    name="UHD Graphics 630",
    vendor=GPUVendor.INTEL,
    driver="i915",
    kernel_modules=["i915"],
    pci_slot="0000:00:02.0",
)

NVIDIA_GPU = _make_gpu(
    name="RTX 2070 SUPER",
    vendor=GPUVendor.NVIDIA,
    driver="nvidia",
    kernel_modules=["nvidia"],
    pci_slot="0000:01:00.0",
)

AMD_NAVI_GPU = _make_gpu(
    name="Navi 23 [Radeon RX 6600]",
    vendor=GPUVendor.AMD,
    driver="amdgpu",
    kernel_modules=["amdgpu"],
    pci_slot="0000:01:00.0",
)

AMD_ELLESMERE_GPU = _make_gpu(
    name="Ellesmere [Radeon RX 570/580]",
    vendor=GPUVendor.AMD,
    driver="amdgpu",
    kernel_modules=["amdgpu"],
    pci_slot="0000:01:00.0",
)

AMD_RADEON_DRIVER_GPU = _make_gpu(
    name="Ellesmere [Radeon RX 570/580]",
    vendor=GPUVendor.AMD,
    driver="radeon",
    kernel_modules=["radeon", "amdgpu"],
    pci_slot="0000:01:00.0",
)

ROC_PLATFORM = OpenCLPlatform(
    name="AMD Accelerated Parallel Processing",
    icd_suffix="AMD",
    extensions="cl_khr_icd cl_khr_byte_addressable_store",
    devices=[OpenCLDevice(name="gfx1032", board_name="AMD Radeon RX 6600")],
)

NVIDIA_CL_PLATFORM = OpenCLPlatform(
    name="NVIDIA CUDA",
    icd_suffix="NV",
    extensions="cl_khr_icd",
    devices=[OpenCLDevice(name="NVIDIA GeForce RTX 2070 SUPER")],
)

CLOVER_PLATFORM = OpenCLPlatform(
    name="Clover",
    icd_suffix="MESA",
    extensions="cl_khr_icd",
    devices=[OpenCLDevice(name="AMD Radeon RX 6600")],
)

POCL_PLATFORM = OpenCLPlatform(
    name="Portable Computing Language",
    icd_suffix="POCL",
    extensions="cl_khr_icd",
    devices=[OpenCLDevice(name="pthread-skylake-avx512-Intel(R) Core(TM)")],
)


@pytest.fixture()
def nvidia_desktop() -> SystemState:
    return _make_state(
        gpus=[NVIDIA_GPU],
        opencl_drivers=["opencl-nvidia"],
        opencl_platforms=[NVIDIA_CL_PLATFORM],
        opencl_nvidia_installed=True,
        gl_vendor="NVIDIA Corporation",
        gl_renderer="NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2",
        installed_dr_package="davinci-resolve 19.0-1",
    )


@pytest.fixture()
def optimus_laptop() -> SystemState:
    return _make_state(
        chassis=ChassisType.LAPTOP,
        gpus=[INTEL_GPU, NVIDIA_GPU],
        opencl_drivers=["opencl-nvidia"],
        opencl_platforms=[NVIDIA_CL_PLATFORM],
        opencl_nvidia_installed=True,
        gl_vendor="NVIDIA Corporation",
        gl_renderer="NVIDIA GeForce RTX 2070 SUPER/PCIe/SSE2",
    )


@pytest.fixture()
def amd_desktop() -> SystemState:
    return _make_state(
        gpus=[AMD_NAVI_GPU],
        opencl_drivers=["rocm-opencl-runtime"],
        opencl_platforms=[ROC_PLATFORM],
        gl_vendor="AMD",
        gl_renderer="AMD Radeon RX 6600 (navi23, LLVM 17.0.6, DRM 3.54)",
    )


@pytest.fixture()
def amd_pre_vega() -> SystemState:
    return _make_state(
        gpus=[AMD_ELLESMERE_GPU],
        opencl_drivers=["rocm-opencl-runtime"],
        opencl_platforms=[ROC_PLATFORM],
        gl_vendor="AMD",
        gl_renderer="AMD Radeon RX 580 Series (polaris10, LLVM 17.0.6)",
        roc_enable_pre_vega=True,
    )


@pytest.fixture()
def intel_only() -> SystemState:
    return _make_state(
        gpus=[INTEL_GPU],
        gl_vendor="Intel",
        gl_renderer="Mesa Intel(R) UHD Graphics 630",
    )
