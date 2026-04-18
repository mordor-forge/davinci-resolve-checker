from __future__ import annotations

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel


class GPUVendor(str, Enum):
    INTEL = "Intel Corporation"
    AMD = "Advanced Micro Devices, Inc. [AMD/ATI]"
    NVIDIA = "NVIDIA Corporation"


class ChassisType(str, Enum):
    OTHER = "Other"
    UNKNOWN = "Unknown"
    DESKTOP = "Desktop"
    LOW_PROFILE_DESKTOP = "Low Profile Desktop"
    PIZZA_BOX = "Pizza Box"
    MINI_TOWER = "Mini Tower"
    TOWER = "Tower"
    PORTABLE = "Portable"
    LAPTOP = "Laptop"
    NOTEBOOK = "Notebook"
    HAND_HELD = "Hand Held"
    DOCKING_STATION = "Docking Station"
    ALL_IN_ONE = "All in One"
    SUB_NOTEBOOK = "Sub Notebook"
    SPACE_SAVING = "Space-saving"
    LUNCH_BOX = "Lunch Box"
    MAIN_SERVER_CHASSIS = "Main Server Chassis"
    EXPANSION_CHASSIS = "Expansion Chassis"
    SUBCHASSIS = "SubChassis"
    BUS_EXPANSION_CHASSIS = "Bus Expansion Chassis"
    PERIPHERAL_CHASSIS = "Peripheral Chassis"
    RAID_CHASSIS = "RAID Chassis"
    RACK_MOUNT_CHASSIS = "Rack Mount Chassis"
    SEALED_CASE_PC = "Sealed-case PC"
    MULTI_SYSTEM_CHASSIS = "Multi-system chassis"
    COMPACT_PCI = "Compact PCI"
    ADVANCED_TCA = "Advanced TCA"
    BLADE = "Blade"
    BLADE_ENCLOSURE = "Blade Enclosure"
    TABLET = "Tablet"
    CONVERTIBLE = "Convertible"
    DETACHABLE = "Detachable"
    IOT_GATEWAY = "IoT Gateway"
    EMBEDDED_PC = "Embedded PC"
    MINI_PC = "Mini PC"
    STICK_PC = "Stick PC"

    @property
    def is_mobile(self) -> bool:
        return self in {
            ChassisType.LAPTOP,
            ChassisType.NOTEBOOK,
            ChassisType.CONVERTIBLE,
            ChassisType.PORTABLE,
            ChassisType.TABLET,
            ChassisType.DETACHABLE,
            ChassisType.HAND_HELD,
            ChassisType.SUB_NOTEBOOK,
        }

    @property
    def is_desktop(self) -> bool:
        return self in {
            ChassisType.DESKTOP,
            ChassisType.ALL_IN_ONE,
            ChassisType.MINI_PC,
            ChassisType.SPACE_SAVING,
            ChassisType.TOWER,
            ChassisType.OTHER,
        }


class GPUDevice(BaseModel):
    name: str
    vendor: GPUVendor
    driver: str | None
    kernel_modules: list[str]
    pci_slot: str
    pci_class: int

    PRE_VEGA_CODENAMES: ClassVar[list[str]] = ["Ellesmere"]
    VEGA_AND_LATER_CODENAMES: ClassVar[list[str]] = [
        "Vega",
        "Navi",
        "Cezanne",
        "Raphael",
        "Barcelo",
        "Rembrandt",
    ]

    @property
    def is_pre_vega(self) -> bool | None:
        if self.vendor != GPUVendor.AMD:
            return None
        if any(cn in self.name for cn in self.PRE_VEGA_CODENAMES):
            return True
        if any(cn in self.name for cn in self.VEGA_AND_LATER_CODENAMES):
            return False
        return None


class OpenCLDevice(BaseModel):
    name: str
    board_name: str | None = None


class OpenCLPlatform(BaseModel):
    name: str
    icd_suffix: str
    extensions: str
    devices: list[OpenCLDevice]

    NVIDIA_ICD_SUFFIXES: ClassVar[set[str]] = {"NV", "NVIDIA"}

    @property
    def has_devices(self) -> bool:
        return len(self.devices) > 0

    @property
    def is_clover(self) -> bool:
        return self.name == "Clover"

    @property
    def is_orca(self) -> bool:
        return self.icd_suffix == "AMD" and "cl_amd_offline_devices" in self.extensions

    @property
    def is_roc(self) -> bool:
        return self.icd_suffix == "AMD" and not self.is_orca

    @property
    def is_amd(self) -> bool:
        return self.icd_suffix == "AMD"

    @property
    def is_nvidia(self) -> bool:
        return self.icd_suffix in self.NVIDIA_ICD_SUFFIXES or self.name.startswith("NVIDIA")


class SystemState(BaseModel):
    distro_id: str
    distro_name: str
    chassis: ChassisType
    gpus: list[GPUDevice]
    opencl_drivers: list[str]
    opencl_platforms: list[OpenCLPlatform]
    opencl_nvidia_installed: bool
    gl_vendor: str
    gl_renderer: str
    installed_dr_package: str | None
    package_versions: dict[str, str] = {}
    roc_enable_pre_vega: bool = False


class CheckStatus(str, Enum):
    PASS = "pass"  # noqa: S105
    FAIL = "fail"
    WARNING = "warning"


class CheckResult(BaseModel):
    status: CheckStatus
    message: str
    suggestion: str | None = None
