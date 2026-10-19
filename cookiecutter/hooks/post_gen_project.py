"""Copy the pipeline into the new project.

The template itself holds only the project's own files (run overlay,
bait/prey lists, local config, README header). The pipeline -- Snakefile,
src/nanopulldown, envs, report assets, shared config -- is copied here from
the repository the template lives in, as it is checked out (git-tracked files
only), so a new project always gets the pipeline the template came with and
the template never goes stale. `cookiecutter --checkout <ref>` picks another
version of a remote template.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from fnmatch import fnmatch
from pathlib import Path

# relative when cookiecutter was given a relative local path: that is relative to
# where it was invoked ($PWD, inherited), not to this hook's cwd (the new project)
TEMPLATE_DIR = Path("{{ cookiecutter._repo_dir }}")
if not TEMPLATE_DIR.is_absolute():
    TEMPLATE_DIR = Path(os.environ.get("PWD", ".")) / TEMPLATE_DIR
RUN = "{{ cookiecutter.run_name }}"
INIT_GIT = "{{ cookiecutter.init_git }}" == "yes"
AUTHOR, EMAIL = "{{ cookiecutter.author_name }}", "{{ cookiecutter.author_email }}"

# what a new project gets from the pipeline repo (git-tracked files only)
INCLUDE = ["Snakefile", "pyproject.toml", "config.yaml", "config.local.yaml.example", ".gitignore",
           "README.md", "src/*", "envs/*", "report/*", "configs/_schema.md", "configs/curated_domains.yaml"]
# rendered by the template already: never overwritten (README.md is appended to instead)
KEEP = {"config.local.yaml", "configs/protein_names.yaml", "configs/fragment_domains.yaml"}
# tracked, but for the source project's notebooks / slides, which a new project does not get
EXCLUDE = {"envs/esmfold2_notebook.yaml", "envs/presentation.yaml"}


def git(*args, cwd, check=True) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True).stdout.strip()


def blank_pairs(text: str) -> str:
    """The source project's own `pairs:` list -> empty (a run's pairs come from its pairs_file)."""
    out, n = re.subn(r"^pairs:\n(?:[ \t]+\S.*\n)*",
                     "pairs: []      # a run's pairs come from its overlay's pairs_file (inputs expand)\n",
                     text, count=1, flags=re.M)
    if not n:
        print("WARNING: no `pairs:` block found in config.yaml", file=sys.stderr)
    return out


def main() -> int:
    try:
        src = Path(git("rev-parse", "--show-toplevel", cwd=TEMPLATE_DIR))
        files = git("ls-files", cwd=src).splitlines()
        commit = git("rev-parse", "HEAD", cwd=src)
        dirty = bool(git("status", "--porcelain", "--untracked-files=no", cwd=src))
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ERROR: the template must live in a git checkout of the pipeline ({TEMPLATE_DIR}): {e}",
              file=sys.stderr)
        return 1

    dst = Path.cwd()
    copied = 0
    for rel in files:
        if not any(fnmatch(rel, pat) for pat in INCLUDE) or rel in KEEP | EXCLUDE or not (src / rel).is_file():
            continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if rel == "README.md":
            target.write_text(target.read_text() + (src / rel).read_text())
        elif rel == "config.yaml":
            target.write_text(blank_pairs((src / rel).read_text()))
        else:
            shutil.copy2(src / rel, target)
        copied += 1

    (dst / ".nanopulldown-template.json").write_text(json.dumps(
        {"source": str(src), "commit": commit, "uncommitted_changes": dirty}, indent=2) + "\n")
    print(f"[nanopulldown] copied {copied} pipeline file(s) from {src} @ {commit[:10]}"
          + (" (with uncommitted changes)" if dirty else ""))

    if INIT_GIT:
        git("init", "-q", cwd=dst)
        git("add", "-A", cwd=dst)
        # the user's own git identity when set, else the template's author
        has_ident = git("config", "user.email", cwd=dst, check=False)
        ident = [] if has_ident else ["-c", f"user.name={AUTHOR}", "-c", f"user.email={EMAIL}"]
        if subprocess.run(["git", *ident, "commit", "-q", "-m", "New nanopulldown project from template"],
                          cwd=dst, capture_output=True).returncode != 0:
            print("[nanopulldown] git init done; initial commit skipped (commit it yourself)")

    print(f"""
Next steps:
  cd {dst.name}
  conda env create -f envs/controller.yaml && conda activate nanopulldown && pip install -e .
  edit configs/baits_{RUN}.txt / configs/preys_{RUN}.txt and config_{RUN}.yaml as needed
  nanopulldown --run-config config_{RUN}.yaml inputs expand --dry-run
  nanopulldown --run-config config_{RUN}.yaml inputs expand
  NANOPULLDOWN_RUN_CONFIG=config_{RUN}.yaml snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy={{ cookiecutter.docker_heavy_pool }}
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
