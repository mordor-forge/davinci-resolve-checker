from __future__ import annotations

import subprocess

from pylspci.parsers import VerboseParser

from davinci_resolve_checker.models import GPUDevice, GPUVendor

GLXINFO_VENDOR_PREFIX = "OpenGL vendor string:"
GLXINFO_RENDERER_PREFIX = "OpenGL renderer string:"
EGLINFO_VENDOR_RENDERER_PREFIXES = (
    (
        "OpenGL core profile vendor:",
        "OpenGL core profile renderer:",
    ),
    (
        "OpenGL compatibility profile vendor:",
        "OpenGL compatibility profile renderer:",
    ),
    (
        "OpenGL ES profile vendor:",
        "OpenGL ES profile renderer:",
    ),
)


def probe_gpus() -> list[GPUDevice]:
    lspci_devices = VerboseParser().run()
    gpus: list[GPUDevice] = []

    for device in lspci_devices:
        if device.cls.id not in (0x0300, 0x0301, 0x0302, 0x0380):
            continue
        if device.driver == "vfio-pci":
            continue

        try:
            vendor = GPUVendor(device.vendor.name)
        except ValueError:
            continue

        gpus.append(
            GPUDevice(
                name=device.device.name,
                vendor=vendor,
                driver=device.driver,
                kernel_modules=list(device.kernel_modules) if device.kernel_modules else [],
                pci_slot=str(device.slot),
                pci_class=device.cls.id,
            )
        )

    return gpus


def _run_graphics_probe(command: list[str]) -> str:
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    return result.stdout


def _extract_prefixed_value(output: str, prefix: str) -> str:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return stripped.removeprefix(prefix).strip()

    return ""


def probe_gl_info() -> tuple[str, str]:
    glxinfo_output = _run_graphics_probe(["glxinfo", "-B"])
    vendor = _extract_prefixed_value(glxinfo_output, GLXINFO_VENDOR_PREFIX)
    renderer = _extract_prefixed_value(glxinfo_output, GLXINFO_RENDERER_PREFIX)
    if vendor and renderer:
        return vendor, renderer

    eglinfo_output = _run_graphics_probe(["eglinfo", "-B"])
    for vendor_prefix, renderer_prefix in EGLINFO_VENDOR_RENDERER_PREFIXES:
        vendor = _extract_prefixed_value(eglinfo_output, vendor_prefix)
        renderer = _extract_prefixed_value(eglinfo_output, renderer_prefix)
        if vendor and renderer:
            return vendor, renderer

    return "", ""
