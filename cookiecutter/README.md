# nanopulldown project template

Starts a new bait x prey screen with this pipeline:

```bash
pipx install cookiecutter        # or: pip install cookiecutter / conda install -c conda-forge cookiecutter
cookiecutter gh:EtienneReboul/NanoPulldown --directory cookiecutter
# or from a local checkout:
cookiecutter ~/projects/NanoPulldown/cookiecutter
```

It asks for a project name, the baits and preys (AlphaPulldown specs,
comma-separated), the organism, and whether to use fragment mode, positive
controls and decoys. The new project gets:

| From the answers (`{{cookiecutter.project_slug}}/`) | Copied from the pipeline repo (hook)                  |
| --------------------------------------------------- | ----------------------------------------------------- |
| `config_<run>.yaml` (the run overlay)               | `Snakefile`, `pyproject.toml`, `src/nanopulldown/`    |
| `configs/baits_<run>.txt`, `configs/preys_<run>.txt` | `config.yaml` (with the source project's `pairs:` emptied) |
| `config.local.yaml` (your InterProScan e-mail)      | `envs/`, `report/`, `configs/_schema.md`, `configs/curated_domains.yaml` |
| `configs/protein_names.yaml`, `configs/fragment_domains.yaml` (empty) | `.gitignore`, the pipeline README (appended) |
| README header                                       | `.nanopulldown-template.json`: source path + commit   |

The pipeline is copied by `hooks/post_gen_project.py` from the git checkout
the template lives in (tracked files only), so the template never holds a
second copy of the code. A project gets the pipeline at that checkout's
commit; `cookiecutter --checkout <tag-or-branch>` picks another one from
GitHub. Nothing the source project ran (its run overlays, pair lists,
results, notebooks, slides) is copied.

Answers are validated first (`hooks/pre_gen_project.py`): lowercase
slug/run name, whole numbers, well-formed protein specs.
