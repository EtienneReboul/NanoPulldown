"""
nanopulldown.core.containers
============================
One place that turns "run this tool image" into a command line, for the
two tools that ship as container images (PLIP, AIUPred), under config.yaml's
`execution.container_runtime`:

  docker     `docker run --rm ...` -- the Mac (Docker Desktop).
  apptainer  `apptainer run <images_dir>/<name>.sif ...` -- the IFB cluster
             (no docker daemon there). `apptainer run` executes the image's
             ENTRYPOINT with the arguments, like `docker run` does.

The rule envs themselves are conda envs on both machines; only these
third-party images go through here.

Stdlib only.
"""
from __future__ import annotations

from pathlib import Path


def runtime(cfg: dict) -> str:
    return (cfg.get("execution") or {}).get("container_runtime", "docker")


def sif_path(cfg: dict, name: str) -> Path:
    """<images_dir>/<name>.sif -- as scripts/ifb/setup.sh pulls them."""
    images_dir = (cfg.get("execution") or {}).get("images_dir") or ""
    if not images_dir:
        raise ValueError("execution.images_dir is not set (needed with container_runtime: apptainer)")
    return Path(images_dir) / f"{name}.sif"


def run_command(cfg: dict, *, image: str, sif: str, args: list[str],
                mounts: dict[str, str] | None = None, workdir: str | None = None,
                platform: str = "", memory: str = "", gpu: bool = False,
                entrypoint: str | None = None) -> list[str]:
    """argv running `image` (docker) / <images_dir>/<sif>.sif (apptainer) with `args`.
    `mounts` = {host: container}; memory/platform only apply to docker
    (apptainer runs on the job's own SLURM allocation, natively amd64)."""
    mounts = mounts or {}
    if runtime(cfg) == "apptainer":
        cmd = ["apptainer", "exec" if entrypoint else "run", "--cleanenv"]
        cmd += (cfg.get("execution") or {}).get("apptainer_args", "").split()
        for host, ctr in mounts.items():
            cmd += ["--bind", f"{host}:{ctr}"]
        if workdir:
            cmd += ["--pwd", workdir]
        if gpu:
            cmd.append("--nv")
        cmd.append(str(sif_path(cfg, sif)))
        return cmd + ([entrypoint] if entrypoint else []) + args
    cmd = ["docker", "run", "--rm"]
    if platform:
        cmd += ["--platform", platform]
    if memory:
        cmd.append(f"--memory={memory}")
    if gpu:
        cmd += ["--gpus", "all"]
    for host, ctr in mounts.items():
        cmd += ["-v", f"{host}:{ctr}"]
    if workdir:
        cmd += ["-w", workdir]
    if entrypoint:
        cmd += ["--entrypoint", entrypoint]
    return cmd + [image] + args


def aiupred_command(cfg: dict, acfg: dict, workdir: Path, aiupred_args: list[str]) -> tuple[list[str], dict]:
    """(argv, subprocess kwargs) running `python3 -m aiupred.cli <aiupred_args>`
    in the AIUPred image, with `workdir` (host) mounted at /work. The image
    sets no ENTRYPOINT and runs from WORKDIR /opt/aiupred, hence the explicit
    interpreter (and, on apptainer, --pwd: it starts in the host cwd)."""
    gpu = bool(acfg.get("use_gpu"))
    image = acfg["image_gpu"] if gpu else acfg["image_cpu"]
    sif = "aiupred_gpu" if gpu else "aiupred"
    return run_command(cfg, image=image, sif=sif, args=["-m", "aiupred.cli", *aiupred_args],
                       mounts={str(workdir): "/work"}, platform=acfg.get("docker_platform", ""),
                       gpu=gpu, entrypoint="python3",
                       workdir="/opt/aiupred" if runtime(cfg) == "apptainer" else None), {}
