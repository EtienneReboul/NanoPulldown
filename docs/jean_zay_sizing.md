# Resource estimate: NanoPulldown on Jean Zay `gpu_p6` (H100 SXM5 80 GB) and Adastra (MI250X)

Target: **1,800 bait × prey pairs per year**, each folded with ESMFold2-Fast into
**25 structures** (5 seeds × 5 diffusion samples; 20 loops, 200 sampling steps,
`lm_dropout` 0.3), followed by OpenMM minimization, PLIP contacts and the report.

All numbers below were **measured on IFB's H200 NVL (141 GB)** in October 2026 and
translated to the H100. Nothing has been run on an H100 yet.

## 1. Measured per-run cost (H200, CUDA, torch 2.11 + esm 3.4.1)

One *run* = one seed = one trunk pass + its diffusion samples.

| Pair tokens | 1 sample: time / peak GPU memory | 5 samples: time / peak GPU memory |
| --- | --- | --- |
| 799   | 116 s / 21.8 GiB   | 123 s (+6 %) / 46.7 GiB |
| 1,298 | 407 s / 36.8 GiB   | 477 s (+17 %) / 102.5 GiB |
| 1,800 | 1,019 s / 59.1 GiB | out of memory (reached 137.8 GiB) |
| 2,301 | 2,129 s / 88.6 GiB | out of memory |

Fitted laws (n = total tokens of the pair):

- time per run, 1 sample: **t₁ ≈ 1.24·10⁻⁶ · n^2.74 s** (also matches 41 DNE1 pairs at 700–2,200 tokens);
- extra samples cost **≈ +4 % time each**, but **+9.75·10⁻⁶ · n² GiB memory each**;
- peak memory, 1 sample: **12.6 GiB (weights) + 1.51·10⁻⁵ · n² GiB**.

Diffusion samples are cheap in time but expensive in memory: each one carries its
own copy of the pair representation (n²) through the diffusion and confidence heads.

## 2. What fits on an H100 80 GB

Taking ~73 GiB as usable:

| Pair tokens | Samples per run that fit | Runs for 25 structures | GPU-h per pair (H200 time) |
| --- | --- | --- | --- |
| ≤ 1,000 | 5 | 5  | ≤ 0.34 |
| 1,100   | 4 | 7  | 0.59 |
| 1,300   | 3 | 9  | 1.15 |
| 1,600   | 1 | 25 | 5.3 |
| 1,800   | 1 | 25 | 7.3 |
| 2,000   | 1 | 25 | 9.7 (at the memory limit) |
| > 2,000 | — | — | **does not fit, even at 1 sample** |

Above ~1,500 tokens, the 25 structures cost 25 full trunk passes.

## 3. Estimate for 1,800 pairs × 25 structures

H100 speed relative to the H200 NVL: **1.0–1.4×** the H200 time. Peak compute is
similar, and the H100's memory bandwidth is lower (3.35 vs 4.8 TB/s).

| Screen type (size distribution from real screens) | H100 GPU-hours | + 25 % margin |
| --- | --- | --- |
| **Full-length** (like DNE1 full: median 1,320 tokens, 80 % ≤ 2,000) | 4,840 – 6,780 | **≈ 8,500 GPU-h** (≈ 88 node-days) |
| **Fragment / domain** (like DNE1 fragments: median 388, all ≤ 1,100) | 74 – 103 | **≈ 130 GPU-h** |

For comparison, keeping **5 seeds × 1 sample** (5 structures per pair, the IFB
setting) would cost **≈ 3,300 GPU-h** with margin for the full-length case. That
estimate is scaled from DNE1's measured 524 H200-h for 509 pairs.

Other needs, per 1,800 pairs × 25 structures:

- **Files:** ~0.5 M (structures, minimized models, PLIP reports, logs, workflow
  metadata). The conda environments add ~150 k more, unless they ship as one
  container image each.
- **Disk:** ~0.1 TB (full-length) or ~0.02 TB (fragments), excluding the 25 GB of
  model weights.
- **CPU:** ~45,000 PLIP runs at ~50 s ≈ 600 CPU-h; minimization runs on the GPU in
  seconds per structure.
- **Per node:** 4 independent fold jobs, one per GPU (one process per GPU, no
  multi-GPU parallelism). Each job needs ~100–160 GB of host RAM while loading,
  which fits within 512 GB for 4 jobs.

## 4. Alternative: AMD MI250X on Adastra (CINES; ROCm 6.4, PyTorch ≥ 2.9)

An MI250X card is **two GPU dies (GCDs)**, and PyTorch sees each one as a separate
64 GB device. ESMFold2 runs on a single device: nothing splits one fold across
GCDs. So the memory limit per job is **64 GB (~58 GiB usable)**, not the card's 128 GB.
Spec sheets quote the card (128 GB, 3.3 TB/s, ~48 TFLOPS FP32, ~383 TFLOPS bf16
matrix): halve them for one GCD. The MI250X has no mode that joins its two GCDs into
one device. A card runs two fold jobs side by side, one per GCD.

| Per device | MI250X GCD | H100 SXM5 | H200 NVL (measured) |
| --- | --- | --- | --- |
| Memory | 64 GB | 80 GB | 141 GB |
| Memory bandwidth | 1.6 TB/s | 3.35 TB/s | 4.8 TB/s |
| Peak dense bf16 matrix | ~192 TFLOPS | ~989 | ~835 |

A GCD is expected to be **3–5× slower than the H200** for this workload: ~3× if
bandwidth-bound, ~4.4× if compute-bound. This is from AMD's published specs, not
measured, and before any ROCm software overhead.

What fits on one GCD (same measured memory laws):

| | MI250X GCD | H100 |
| --- | --- | --- |
| 1 sample per run fits up to | ~1,730 tokens | ~2,000 tokens |
| 5 samples per run fit up to | ~915 tokens | ~1,060 tokens |
| DNE1 full-length pairs that fit at all | 61 % | 80 % |
| Fragment pairs that fit | 100 % | 100 % |

Cost of 1,800 pairs (pairs that fit only), in GCD-hours (2 GCDs = 1 card):

| Screen type | 25 structures / pair | 5 structures / pair (5 seeds × 1) |
| --- | --- | --- |
| Full-length (DNE1-like) | 6,550–10,900 GCD-h; **≈ 13,600 with margin** (≈ 6,800 card-h) | 1,390–2,310; **≈ 2,900 with margin** |
| Fragment (DNE1-fragments-like) | 230–380; **≈ 470 with margin** | 47–79; **≈ 100 with margin** |

Software points, all untested on ROCm:

- The fold code's torch backend should run unchanged: on ROCm, `torch.cuda` is HIP
  (the device is still `"cuda"`), and the MI250X supports bf16 autocast. It needs a
  ROCm env (torch from `https://download.pytorch.org/whl/rocm6.4`) in place of
  `envs/esmfold2_cuda.yaml`.
- The NVIDIA-only kernels (cuequivariance, Transformer Engine) are already unused
  on IFB. Triton and flash-attention have ROCm builds, but this workload has not
  been tried with them.
- Minimization: OpenMM has a HIP platform plugin, and OpenCL also works on ROCm.
  Neither has been tried here.

**Fit for this workload:** good for fragment screens and pairs up to ~900 tokens
(5 samples per run). For full-length screens, 39 % of DNE1's pairs do not fit a GCD,
and the rest run 3–5× slower than on an H100, so the H100 partition is the better
match for full-length work.

## 5. How many DNE1-sized pulldowns fit in 10,000 GPU-hours

One pulldown = the DNE1/DCP1/DCP2 IP at full length: **5 baits × (112 screen preys +
5 positive-control preys + n decoy preys)**, every combination, with decoys at
n = 200 (as in the DNE1 fragment run) or n = 500 (config.yaml's default). Decoys are
drawn in length bins matching the preys, so their pairs are assumed to follow the
screen's size distribution. Pairs too large for the device are left out: **30 %** of
the combinations on an H100, **46 %** on an MI250X GCD (both above 2,400 tokens and
above the memory limit).

GPU-hours per pulldown, with a 25 % margin (H100 = one GPU; MI250X = one GCD):

| Device | Structures / pair | Decoys | Pairs | GPU-h per pulldown | Pulldowns in 10,000 GPU-h |
| --- | --- | --- | --- | --- | --- |
| H100 80 GB (Jean Zay) | 25 (5 seeds × 5) | 200 | 1,585 | 3,700 – 5,100 | **≈ 2 – 2.7** |
| H100 80 GB (Jean Zay) | 25 | 500 | 3,085 | 7,200 – 10,000 | **≈ 1 – 1.4** |
| H100 80 GB (Jean Zay) | 5 (5 seeds × 1) | 200 | 1,585 | 750 – 1,050 | **≈ 9.5 – 13** |
| H100 80 GB (Jean Zay) | 5 | 500 | 3,085 | 1,500 – 2,000 | **≈ 5 – 7** |
| MI250X GCD (Adastra) | 25 | 200 | 1,585 | 3,900 – 6,400 | **≈ 1.6 – 2.6** |
| MI250X GCD (Adastra) | 25 | 500 | 3,085 | 7,500 – 12,500 | **≈ 0.8 – 1.3** |
| MI250X GCD (Adastra) | 5 | 200 | 1,585 | 800 – 1,400 | **≈ 7 – 12** |
| MI250X GCD (Adastra) | 5 | 500 | 3,085 | 1,600 – 2,600 | **≈ 4 – 6** |

The ranges are the H100 at 1.0–1.4× and the GCD at 3–5× the measured H200 time.
The MI250X rows look close to the H100 only because 46 % of the pairs (the largest,
most expensive ones) are dropped on a GCD. Per pair folded, a GCD is ~3× more expensive.
Check how Adastra counts a "GPU-hour" (per GCD or per card) before comparing budgets.

**Take-away for 10,000 GPU-hours:** with 25 structures per pair and decoy controls,
**about 2 DNE1-sized pulldowns** (200 decoys). With 5 structures per pair (5 seeds × 1
sample), **about 10–13** on H100s. Decoys dominate the cost: 200 decoys are 63 % of
a pulldown's pairs.

## 6. Caveats and lessons from IFB

- **Pairs above ~2,000 tokens cannot run on an 80 GB H100** without memory work:
  20 % of DNE1's full-length pairs. They need either smaller inputs
  (fragments), a 141 GB GPU, or code that runs the diffusion samples one at a time
  after a single trunk pass. That code doesn't exist yet; it would also cut the
  1,500–2,000-token cost by ~5×.
- **The estimate assumes the measured CUDA path:** plain PyTorch kernels. The
  optimized cuequivariance kernels didn't load (CUDA 13 runtime missing), and
  flash-attn and Transformer Engine aren't installed. A tuned install could be
  noticeably faster.
- **Startup cost:** on network file systems, importing the CUDA env and reading the
  weights took 10–25 min on a cold node (vs 33 s warm). It is negligible for
  multi-hour fold jobs but argues for containers or node-local copies.
- **File-count quotas:** IFB's 500 k-inode project quota was the binding
  constraint, not disk space. Check Jean Zay's `$WORK` / `$SCRATCH` inode quotas
  against the ~0.5–0.65 M files above.
