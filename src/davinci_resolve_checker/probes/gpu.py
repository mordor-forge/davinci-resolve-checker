from __future__ import annotations

import subprocess

from pylspci.parsers import VerboseParser

from davinci_resolve_checker.models import GPUDevice, GPUVendor


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


def probe_gl_info() -> tuple[str, str]:
    try:
        vendor_result = subprocess.run(
            'glxinfo | grep "OpenGL vendor string" | cut -f2 -d":"',
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        renderer_result = subprocess.run(
            "glxinfo | grep -i 'OpenGL renderer' | cut -f2 -d ':' | xargs",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        vendor = vendor_result.stdout.strip() if vendor_result.returncode == 0 else ""
        renderer = renderer_result.stdout.strip() if renderer_result.returncode == 0 else ""
    except Exception:
        vendor, renderer = "", ""

    return vendor, renderer
