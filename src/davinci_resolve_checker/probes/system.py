from __future__ import annotations

import os
import subprocess

import distro

from davinci_resolve_checker.models import ChassisType

CHASSIS_CODE_MAP: dict[str, ChassisType] = {
    "1": ChassisType.OTHER,
    "2": ChassisType.UNKNOWN,
    "3": ChassisType.DESKTOP,
    "4": ChassisType.LOW_PROFILE_DESKTOP,
    "5": ChassisType.PIZZA_BOX,
    "6": ChassisType.MINI_TOWER,
    "7": ChassisType.TOWER,
    "8": ChassisType.PORTABLE,
    "9": ChassisType.LAPTOP,
    "10": ChassisType.NOTEBOOK,
    "11": ChassisType.HAND_HELD,
    "12": ChassisType.DOCKING_STATION,
    "13": ChassisType.ALL_IN_ONE,
    "14": ChassisType.SUB_NOTEBOOK,
    "15": ChassisType.SPACE_SAVING,
    "16": ChassisType.LUNCH_BOX,
    "17": ChassisType.MAIN_SERVER_CHASSIS,
    "18": ChassisType.EXPANSION_CHASSIS,
    "19": ChassisType.SUBCHASSIS,
    "20": ChassisType.BUS_EXPANSION_CHASSIS,
    "21": ChassisType.PERIPHERAL_CHASSIS,
    "22": ChassisType.RAID_CHASSIS,
    "23": ChassisType.RACK_MOUNT_CHASSIS,
    "24": ChassisType.SEALED_CASE_PC,
    "25": ChassisType.MULTI_SYSTEM_CHASSIS,
    "26": ChassisType.COMPACT_PCI,
    "27": ChassisType.ADVANCED_TCA,
    "28": ChassisType.BLADE,
    "29": ChassisType.BLADE_ENCLOSURE,
    "30": ChassisType.TABLET,
    "31": ChassisType.CONVERTIBLE,
    "32": ChassisType.DETACHABLE,
    "33": ChassisType.IOT_GATEWAY,
    "34": ChassisType.EMBEDDED_PC,
    "35": ChassisType.MINI_PC,
    "36": ChassisType.STICK_PC,
}


def probe_chassis() -> ChassisType:
    with open("/sys/class/dmi/id/chassis_type") as f:
        code = f.read().strip()
    return CHASSIS_CODE_MAP.get(code, ChassisType.UNKNOWN)


def probe_distro() -> tuple[str, str]:
    return distro.id(), distro.name()


def probe_opencl_drivers() -> list[str]:
    result = subprocess.run(
        "expac -Qs '%n' opencl-driver",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [line.strip() for line in result.stdout.strip().splitlines()]


def probe_opencl_nvidia_installed() -> bool:
    result = subprocess.run(
        "expac -Qs '%n' opencl-nvidia",
        shell=True,
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def probe_installed_dr_package() -> str | None:
    result = subprocess.run(
        "expac -Qs '%n %v' davinci-resolve",
        shell=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    return output if output else None


def probe_package_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in packages:
        result = subprocess.run(
            f"expac -Q '%v' {pkg}",
            shell=True,
            capture_output=True,
            text=True,
        )
        version = result.stdout.strip()
        if version:
            versions[pkg] = version
    return versions


def probe_roc_enable_pre_vega() -> bool:
    return os.environ.get("ROC_ENABLE_PRE_VEGA", "0") == "1"
