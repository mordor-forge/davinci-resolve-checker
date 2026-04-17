from __future__ import annotations

import json
import subprocess

from davinci_resolve_checker.models import OpenCLDevice, OpenCLPlatform


def probe_opencl_platforms() -> list[OpenCLPlatform]:
    try:
        result = subprocess.run(
            "clinfo --json",
            shell=True,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return []

        data = json.loads(result.stdout.strip())
    except (json.JSONDecodeError, Exception):
        return []

    platforms: list[OpenCLPlatform] = []

    for i, platform_data in enumerate(data.get("platforms", [])):
        icd_suffix = platform_data.get("CL_PLATFORM_ICD_SUFFIX_KHR", "")

        devices: list[OpenCLDevice] = []
        if i < len(data.get("devices", [])):
            for dev in data["devices"][i].get("online", []):
                board_name = dev.get("CL_DEVICE_BOARD_NAME_AMD")
                name = board_name or dev.get("CL_DEVICE_NAME", "Unknown")
                devices.append(OpenCLDevice(name=name, board_name=board_name))

        platforms.append(
            OpenCLPlatform(
                name=platform_data.get("CL_PLATFORM_NAME", "Unknown"),
                icd_suffix=icd_suffix,
                extensions=platform_data.get("CL_PLATFORM_EXTENSIONS", ""),
                devices=devices,
            )
        )

    return platforms
