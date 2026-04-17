from __future__ import annotations

import os
import subprocess

import distro

from davinci_resolve_checker.models import ChassisType

CHASSIS_CODE_MAP: dict[str, ChassisType] = {
    str(i + 1): ct for i, ct in enumerate(ChassisType)
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
