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

AIUPred is the exception on apptainer: its caller (annotate disorder) already
runs inside the analysis image, which is built FROM the AIUPred image
(containers/Dockerfile), so AIUPred is called directly, not nested --
see aiupred_command().

Stdlib only.
"""
from __future__ import annotations

from pathlib import Path


def runtime(cfg: dict) -> str:
    return (cfg.get("execution") or {}).get("container_runtime", "docker")


def sif_path(cfg: dict, name: str) -> Path:
    """<images_dir>/<name>.sif -- names as scripts/ifb/build_images.sh writes them."""
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


# The AIUPred image's own interpreter and package dir (ghcr.io/doszilab/aiupred,
# Debian, python 3.11; WORKDIR /opt/aiupred). The analysis image keeps them
# intact underneath its micromamba env.
AIUPRED_PYTHON = "/usr/local/bin/python3"
AIUPRED_DIR = "/opt/aiupred"


def aiupred_command(cfg: dict, acfg: dict, workdir: Path, aiupred_args: list[str]) -> tuple[list[str], dict]:
    """(argv, subprocess kwargs) running `python3 -m aiupred.cli <aiupred_args>`
    with `workdir` (host) visible as /work. On apptainer it runs directly:
    we are already inside the analysis image, so /work is just `workdir`."""
    gpu = bool(acfg.get("use_gpu"))
    if runtime(cfg) == "apptainer":
        args = [a.replace("/work", str(workdir)) for a in aiupred_args]
        env = {"PYTHONPATH": AIUPRED_DIR, "PATH": "/usr/local/bin:/usr/bin:/bin"}
        return [AIUPRED_PYTHON, "-m", "aiupred.cli", *args], {"cwd": AIUPRED_DIR, "env": env}
    image = acfg["image_gpu"] if gpu else acfg["image_cpu"]
    cmd = run_command(cfg, image=image, sif="", args=["-m", "aiupred.cli", *aiupred_args],
                      mounts={str(workdir): "/work"}, platform=acfg.get("docker_platform", ""),
                      gpu=gpu, entrypoint="python3")
    return cmd, {}
