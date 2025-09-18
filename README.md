# MeanFlowSE — One‑Step Generative Speech Enhancement

> **MeanFlowSE** is a conditional generative model that learns **average velocity** over finite time spans and enables **single‑step (1‑NFE) inference** for speech enhancement—no teacher models or distillation required. The method instantiates the Mean Flow identity with a **Jacobian–vector product (JVP)** objective while remaining **exactly consistent** with conditional flow matching on the diagonal. See the paper for details and results.&#x20;
![Uploading image.png…]()

---

## Highlights

* **1‑step inference (Euler‑MF)**: Replace multi‑step ODE integration with a single backward‑in‑time displacement.
* **Local, teacher‑free objective**: MeanFlow identity + JVP target; reduces to CFM at `r=t`.&#x20;
* **Few‑step refinement**: Optional multi‑step variant (Euler along the instantaneous field) for extra quality.
* **Competitive quality, low RTF**: On VoiceBank‑DEMAND, the 1‑step model achieves strong ESTOI, SI‑SDR and DNSMOS with **lowest RTF**, matching or surpassing multi‑step baselines.&#x20;

---

## Method (at a glance)

* **Path**: linear–Gaussian interpolation in complex STFT,
  $\mu_t=(1-t)\,x_1+t\,y,\ \sigma_t=(1-t)\sigma_{\min}+t\sigma_{\max}$, $t\in[0,1]$.&#x20;
* **Targets**:

  * *Instantaneous (CFM)* along the path using the closed form $v_t$.
  * *Mean Flow* target via the identity
    $u = v - c\,(t-r)\,\big(v\cdot\nabla_x u + \partial_t u\big)$ with a stabilizing **first‑order coefficient** $c=\mathbf{0.5}$ (default; set $c=1$ to recover the pure identity).&#x20;
* **One‑step sampler** (Euler‑MF):
  $x_{r}=x_{t}-(t-r)\,u_\theta(x_t,r,t\,|\,y)$.
  For strict 1‑NFE from $t=1\to0$, use `t_eps=0`.&#x20;

---

## Repository structure

```
MeanFLowSE/
├── evaluate.py
├── flowmse
│   ├── backbones
│   │   ├── __init__.py
│   │   ├── dcunet.py
│   │   ├── ncsnpp.py
│   │   ├── ncsnpp_utils
│   │   │   ├── layers.py
│   │   │   ├── layerspp.py
│   │   │   ├── normalization.py
│   │   │   ├── op
│   │   │   │   ├── __init__.py
│   │   │   │   ├── fused_act.py
│   │   │   │   ├── fused_bias_act.cpp
│   │   │   │   ├── fused_bias_act_kernel.cu
│   │   │   │   ├── upfirdn2d.cpp
│   │   │   │   ├── upfirdn2d.py
│   │   │   │   └── upfirdn2d_kernel.cu
│   │   │   ├── up_or_down_sampling.py
│   │   │   └── utils.py
│   │   └── shared.py
│   ├── data_module.py
│   ├── drift_diffusion.py
│   ├── model.py
│   ├── odes.py
│   ├── sampling
│   │   ├── __init__.py
│   │   └── odesolvers.py
│   ├── scripts
│   │   └── train_vbd.sh
│   └── util
│       ├── inference.py
│       ├── other.py
│       ├── registry.py
│       └── tensors.py
├── requirements.txt
├── run_inference.sh
├── train.py
└── utils.py
```

---

## Installation

```bash
# Python 3.10 (recommended)
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt
```

> **Note**: Use a recent PyTorch + CUDA build for multi‑GPU training.

---

## Data preparation

Organize your dataset as:

```
<BASE_DIR>/
  train/clean/*.wav   train/noisy/*.wav
  valid/clean/*.wav   valid/noisy/*.wav
  test/clean/*.wav    test/noisy/*.wav
```

Defaults assume **16 kHz**, Hann window, centered frames, and the complex STFT mapping $|z|^{0.5} e^{j\angle z}$ with scale `0.15`. (See `SpecsDataModule` for options.)

---

## Training

**Single machine, multi‑GPU (DDP)**:

```bash
sh train_vbd.sh
```

**Logging & checkpoints**

* TensorBoard events and checkpoints live under
  `lightning_logs/<exp_name>/version_x/` and
  `lightning_logs/<exp_name>/version_x/checkpoints/`.
* Heavy validation metrics (PESQ/ESTOI/SI‑SDR) are **rank‑0 only** and run **every N epochs** (`--val_metrics_every_n_epochs`, default 20).
  In non‑evaluation epochs, placeholder metrics are logged to keep checkpoint monitors well-defined.

---

## Inference

`evaluate.py` performs enhancement and **writes only the enhanced .wav files**. It will **auto‑pick** the sampler if not specified: `euler_mf` (displacement) when the model was trained with MF‑SE, otherwise `euler` (instantaneous).
sh run_inference.sh

## Key configuration knobs

* **Path & time**: `--T_rev` (start), `--t_eps` (terminal), `--sigma_min/max` (noise schedule)
* **MF‑SE stability**:
  `--mf_first_order_coef` *(default 0.5)*,
  `--mf_jvp_impl {auto,fd,autograd}`, `--mf_jvp_chunk`, `--mf_jvp_clip`, `--mf_jvp_eps`
* **Curriculum**: `--mf_weight_final`, `--mf_warmup_frac`, `--mf_delta_*`, `--mf_r_equals_t_prob`
* **Validation cost**: `--val_metrics_every_n_epochs`, `--num_eval_files`

---

## Results (summary)

On VoiceBank–DEMAND, **1‑step** MeanFlowSE attains strong intelligibility and fidelity with **lowest real‑time factor**, surpassing or matching multi‑step flows/diffusion under identical front‑end and normalization; see paper Tables/Figures for full metrics (ESTOI, SI‑SDR, DNSMOS, SpkSim, RTF).&#x20;

---

## Acknowledgments

* Implementation style inspired by **SGMSE** (score‑based SE in complex STFT) and **FLOWSE**.
* We build on ideas from **Flow Matching** and **Mean Flows**; see the paper for a concise overview and references.&#x20;

---

## Contact

Questions, issues, or pull requests are welcome. For research inquiries, please contact the corresponding authors listed in the paper.&#x20;
