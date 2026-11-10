#!/usr/bin/env python3
"""
nanopulldown.annotate.deepcoil_runner — runs INSIDE envs/deepcoil.yaml
=======================================================================
DeepCoil 2 (Ludwiczak et al. 2019, Bioinformatics 35:2790; v2: SeqVec
embeddings + heptad register) over a multi-FASTA whose headers are
md5(sequence), filling the per-sequence cache
<annotation>/_cache/deepcoil/<md5>.json:

  {"version": 1, "cc": [...], "raw_cc": [...], "prob_a": [...], "prob_d": [...]}
      one value per residue, 3 decimals: cc = DeepCoil's sharpened
      coiled-coil probability (what its CLI writes as `cc`), raw_cc the
      unsharpened one, prob_a / prob_d the heptad a / d position probabilities
  {"version": 1, "skipped": "<reason>"}
      sequences DeepCoil refuses (< 20 aa), so they are not retried

Standalone on purpose: the deepcoil env is Python 3.7 with nothing of
nanopulldown installed, so this file imports only the standard library,
numpy and deepcoil, and is run by path (`python -I .../deepcoil_runner.py`).
The SeqVec weights come from --weights (rule fetch_deepcoil_weights), not
from deepcoil's own first-use download, which needs internet.

Usage:
  python -I deepcoil_runner.py --fasta in.fa --cache-dir DIR --weights DIR \
      [--n-cpu N] [--chunk K] [--done FILE]
"""
import argparse
import json
import os
import sys
import tempfile
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
VERSION = 1


def read_fasta(path):
    out, name, buf = {}, None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name:
                    out[name] = "".join(buf)
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name:
        out[name] = "".join(buf)
    return out


def is_current(path):
    try:
        with open(path) as fh:
            return json.load(fh).get("version") == VERSION
    except (OSError, ValueError):
        return False


def write_json(path, obj):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def load_model(weights, n_cpu):
    from deepcoil import DeepCoil
    from deepcoil.utils.seqvec import SeqVec

    model_dir = os.path.join(weights, "uniref50_v2")
    for f in ("options.json", "weights.hdf5"):
        if not os.path.isfile(os.path.join(model_dir, f)):
            sys.exit("[deepcoil] missing SeqVec weights {} (rule fetch_deepcoil_weights)".format(
                os.path.join(model_dir, f)))

    class LocalDeepCoil(DeepCoil):
        def _setup_seqvec(self):            # never download: our weights dir, CPU
            return SeqVec(model_dir=model_dir, cuda_device=-1, tokens_per_batch=8000)

    return LocalDeepCoil(use_gpu=False, n_cpu=n_cpu)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--weights", required=True, help="dir holding uniref50_v2/{options.json,weights.hdf5}")
    ap.add_argument("--n-cpu", type=int, default=-1)
    ap.add_argument("--chunk", type=int, default=100, help="sequences per predict() call")
    ap.add_argument("--done", default=None)
    a = ap.parse_args()

    os.makedirs(a.cache_dir, exist_ok=True)
    seqs = read_fasta(a.fasta)
    todo = {}
    for md5, seq in seqs.items():
        path = os.path.join(a.cache_dir, md5 + ".json")
        if is_current(path):
            continue
        if len(seq) < 20:
            write_json(path, {"version": VERSION, "skipped": "shorter than 20 aa"})
        else:
            todo[md5] = seq
    print("[deepcoil] {} unique sequence(s), {} to run".format(len(seqs), len(todo)), flush=True)

    failed = 0
    if todo:
        import numpy as np
        from deepcoil.utils import sharpen_preds

        t0 = time.time()
        dc = load_model(a.weights, a.n_cpu)
        print("[deepcoil] model loaded ({:.0f}s)".format(time.time() - t0), flush=True)
        keys = sorted(todo, key=lambda k: len(todo[k]))      # similar lengths batch together
        done = 0
        for i in range(0, len(keys), a.chunk):
            part = {k: todo[k] for k in keys[i:i + a.chunk]}
            try:
                preds = dc.predict(part)
            except Exception as e:                           # one bad chunk must not cost the run
                print("[deepcoil] chunk {} FAILED: {}".format(i // a.chunk, e), file=sys.stderr, flush=True)
                failed += len(part)
                continue
            for k, p in preds.items():
                raw = np.asarray(p["cc"], dtype=float)
                hept = np.asarray(p["hept"], dtype=float)
                if len(raw) != len(part[k]):
                    print("[deepcoil] {}: {} predictions for {} residues, dropped".format(
                        k, len(raw), len(part[k])), file=sys.stderr)
                    failed += 1
                    continue
                rnd = lambda v: [round(float(x), 3) for x in v]
                write_json(os.path.join(a.cache_dir, k + ".json"),
                           {"version": VERSION, "cc": rnd(sharpen_preds(raw)), "raw_cc": rnd(raw),
                            "prob_a": rnd(hept[:, 1]), "prob_d": rnd(hept[:, 2])})
            failed += len(part) - len(preds)
            done += len(part)
            print("[deepcoil] {}/{} ({:.0f}s)".format(done, len(keys), time.time() - t0), flush=True)

    if failed:
        print("[deepcoil] WARNING: {} sequence(s) without prediction; their chains keep InterPro COILS".format(
            failed), file=sys.stderr)
    if a.done:
        with open(a.done, "w") as fh:
            fh.write("{}/{} cached\n".format(len(seqs) - failed, len(seqs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
