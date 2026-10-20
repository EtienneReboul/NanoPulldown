# One image per rule env (envs/<env>.yaml -> nanopulldown-<env>), for
# `--sdm apptainer` on the IFB cluster: the env is created at /opt/env and put
# first on PATH, so a rule's `python -m nanopulldown ...` runs in it exactly
# as under --use-conda. Built by scripts/ifb/build_images.sh (linux/amd64).
#
#   BASE      image to build on. The analysis env uses the AIUPred image
#             (ghcr.io/doszilab/aiupred:cpu), so `annotate disorder` can call
#             AIUPred's own python (/usr/local/bin/python3, /opt/aiupred)
#             from inside the analysis container instead of nesting one.
#   ENV_FILE  envs/<env>.yaml

ARG BASE=mambaorg/micromamba:2.0.5
FROM mambaorg/micromamba:2.0.5 AS micromamba

FROM ${BASE}
USER root
COPY --from=micromamba /bin/micromamba /usr/local/bin/micromamba
ARG ENV_FILE
COPY ${ENV_FILE} /tmp/env.yaml
ENV MAMBA_ROOT_PREFIX=/opt/micromamba
RUN micromamba create -y -p /opt/env -f /tmp/env.yaml \
 && micromamba clean -afy \
 && rm -rf /root/.cache /tmp/env.yaml
ENV PATH=/opt/env/bin:$PATH
