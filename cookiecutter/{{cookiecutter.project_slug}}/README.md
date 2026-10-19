# {{ cookiecutter.project_name }}

A [nanopulldown](https://github.com/EtienneReboul/NanoPulldown) bait x prey
screen. Author: {{ cookiecutter.author_name }} <{{ cookiecutter.author_email }}>.

| What                      | Where                                               |
| ------------------------- | --------------------------------------------------- |
| run settings              | `config_{{ cookiecutter.run_name }}.yaml` (on top of `config.yaml`) |
| baits / preys             | `configs/baits_{{ cookiecutter.run_name }}.txt`, `configs/preys_{{ cookiecutter.run_name }}.txt` |
| your contact e-mail       | `config.local.yaml` (gitignored)                    |
| outputs                   | `results_{{ cookiecutter.run_name }}/`, `reports_{{ cookiecutter.run_name }}/report.zip` |

## Run it

```bash
conda env create -f envs/controller.yaml && conda activate nanopulldown
pip install -e .

# preview, then build the pairs (needs internet)
nanopulldown --run-config config_{{ cookiecutter.run_name }}.yaml inputs expand --dry-run
nanopulldown --run-config config_{{ cookiecutter.run_name }}.yaml inputs expand

# fold + analyse everything (--resources gpu=1 is REQUIRED: one ~25 GB model at a time)
NANOPULLDOWN_RUN_CONFIG=config_{{ cookiecutter.run_name }}.yaml \
  snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy={{ cookiecutter.docker_heavy_pool }} \
  --keep-going --retries 2 --rerun-incomplete
```

Needs Apple Silicon (ESMFold2-Fast on MLX/Metal) and Docker (AIUPred, PLIP).
The pipeline documentation follows.

---

