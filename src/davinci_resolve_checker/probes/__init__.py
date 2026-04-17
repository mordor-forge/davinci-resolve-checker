from __future__ import annotations

from davinci_resolve_checker.models import SystemState
from davinci_resolve_checker.probes.gpu import probe_gl_info, probe_gpus
from davinci_resolve_checker.probes.opencl import probe_opencl_platforms
from davinci_resolve_checker.probes.system import (
    probe_chassis,
    probe_distro,
    probe_installed_dr_package,
    probe_opencl_drivers,
    probe_opencl_nvidia_installed,
    probe_package_versions,
    probe_roc_enable_pre_vega,
)


def probe_system() -> SystemState:
    distro_id, distro_name = probe_distro()
    gl_vendor, gl_renderer = probe_gl_info()

    return SystemState(
        distro_id=distro_id,
        distro_name=distro_name,
        chassis=probe_chassis(),
        gpus=probe_gpus(),
        opencl_drivers=probe_opencl_drivers(),
        opencl_platforms=probe_opencl_platforms(),
        opencl_nvidia_installed=probe_opencl_nvidia_installed(),
        gl_vendor=gl_vendor,
        gl_renderer=gl_renderer,
        installed_dr_package=probe_installed_dr_package(),
        package_versions=probe_package_versions(["opencl-amd", "amdgpu-pro-libgl"]),
        roc_enable_pre_vega=probe_roc_enable_pre_vega(),
    )
