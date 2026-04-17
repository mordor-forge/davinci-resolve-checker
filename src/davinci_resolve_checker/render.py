from __future__ import annotations

import json

from rich.console import Console
from rich.table import Table

from davinci_resolve_checker import __version__
from davinci_resolve_checker.models import CheckResult, CheckStatus, SystemState

STATUS_STYLES = {
    CheckStatus.PASS: ("PASS", "green"),
    CheckStatus.FAIL: ("FAIL", "red bold"),
    CheckStatus.WARNING: ("WARN", "yellow"),
}


def render_json(state: SystemState, results: list[CheckResult]) -> None:
    output = {
        "system": state.model_dump(mode="json"),
        "results": [r.model_dump(mode="json") for r in results],
    }
    print(json.dumps(output, indent=2))


def render_text(state: SystemState, results: list[CheckResult]) -> None:
    console = Console()

    console.print(f"DaVinci Resolve Checker {__version__}")
    console.print(f"Distro: {state.distro_name} ({state.distro_id})")
    console.print(f"Chassis: {state.chassis.value}")
    if state.installed_dr_package:
        console.print(f"Installed: {state.installed_dr_package}")

    if state.gpus:
        gpu_table = Table(title="GPUs", show_header=True)
        gpu_table.add_column("Device")
        gpu_table.add_column("Driver")
        for gpu in state.gpus:
            gpu_table.add_row(gpu.name, gpu.driver or "-")
        console.print(gpu_table)

    if state.gl_vendor:
        console.print(f"OpenGL vendor: {state.gl_vendor}")
        console.print(f"OpenGL renderer: {state.gl_renderer}")

    if state.opencl_platforms:
        for platform in state.opencl_platforms:
            device_count = len(platform.devices)
            console.print(f"OpenCL: {platform.name} ({device_count} device(s))")

    console.print()

    sorted_results = sorted(results, key=lambda r: list(CheckStatus).index(r.status))

    for result in sorted_results:
        label, style = STATUS_STYLES[result.status]
        console.print(f"[{style}][{label}][/{style}] {result.message}")
        if result.suggestion:
            console.print(f"      → {result.suggestion}")
