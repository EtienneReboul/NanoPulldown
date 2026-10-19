"""Validate the answers before anything is written."""
import re
import sys

slug = "{{ cookiecutter.project_slug }}"
run = "{{ cookiecutter.run_name }}"
baits = "{{ cookiecutter.baits }}"
preys = "{{ cookiecutter.preys }}"
numbers = {"organism_taxon": "{{ cookiecutter.organism_taxon }}", "n_decoys": "{{ cookiecutter.n_decoys }}",
           "max_tokens": "{{ cookiecutter.max_tokens }}", "docker_heavy_pool": "{{ cookiecutter.docker_heavy_pool }}"}

# AlphaPulldown protein spec: ID | ID:N | ID:start-stop[:start-stop...] | ID:N:start-stop[...]
SPEC = re.compile(r"^[A-Za-z0-9_.-]+(:\d+)?(:\d+-\d+)*$")
errors = []
for name, value in (("project_slug", slug), ("run_name", run)):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_]*", value):
        errors.append(f"{name} {value!r}: use lowercase letters, digits and _ only")
for name, value in numbers.items():
    if not value.isdigit():
        errors.append(f"{name} {value!r}: not a whole number")
for role, specs in (("baits", baits), ("preys", preys)):
    items = [s.strip() for s in specs.replace(";", ",").split(",") if s.strip()]
    if role == "baits" and not items:
        errors.append("baits: give at least one bait")
    errors += [f"{role}: {s!r} is not a protein spec (ID, ID:N, ID:start-stop, ID:N:start-stop)"
               for s in items if not SPEC.match(s)]
if errors:
    print("\n".join(f"ERROR: {e}" for e in errors), file=sys.stderr)
    sys.exit(1)
