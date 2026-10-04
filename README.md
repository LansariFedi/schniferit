# schniferit
This app is fully AI coded.

Squeeze any image classifier: smaller, faster, within an accuracy budget you set.

## Install

```bash
pip install -r requirements.txt
```

Use a CUDA-enabled torch build for your platform if you have an NVIDIA GPU.

## Quickstart

Run from the directory containing `schniferit/` (clone it next to your project or add its parent to `PYTHONPATH`).

```bash
# 1. Finetune your classifier and save weights (e.g. artifacts/best.pt)

# 2. See what this machine can run
python -m schniferit list

# 3. Log the baseline (accuracy, size, latency)
python -m schniferit baseline --arch resnet18 --weights artifacts/best.pt

# 4. Run the optimization ladder (compile → int8 → prune → distill)
python -m schniferit optimize --budget 3.0
```

## How it works

Each stage transforms the model, re-measures it, and is **kept or reverted**:

- kept iff accuracy drop ≤ `--budget` (points) **and** latency or size improves
- otherwise reverted with the reason logged — the run never crashes on a failed stage

Every result appends to `results.md`: stage, accuracy, size, CPU/GPU latency, verdict. Rows are never deleted; the history is the product.

## Stages

| # | name | target | effect |
|---|---|---|---|
| 1 | `compile_fp16` | GPU | `torch.compile` + fp16 inference (recipe, no file) |
| 2 | `static_int8` | CPU | post-training static quantization, ~4x smaller |
| 3 | `structured_prune` | CPU | channel pruning + healing finetune (`--prune-ratio`, `--heal-epochs`) |
| 4 | `distill` | CPU | train a small student from the reference (`--student`, `--distill-epochs`) |

Accepted `.pt` artifacts also ship as verified `.onnx` (onnxruntime-checked within 1e-3).

Run a subset with `--stages 1,3`. Stages declare supported tasks; anything else is skipped with a logged reason.

## Layout

```
cli.py               # list / baseline / optimize
measure.py           # size, params, latency, metric
backends.py          # capability probing
models.py            # arch loading (torchvision registry or pkg:factory)
stages.py            # the optimization ladder
export.py            # ONNX export + verification
tasks/classify.py    # recipe pack: loader, metric, transforms
```

## New model families

Add `tasks/<name>.py` with `build_loader`, `default_val_transform`,
`default_train_transform`, and a `metric` function; register it in `TASKS`;
mark stage support in `STAGES`. The core stays untouched.

Custom architectures: `--arch package:factory_fn` (must return `nn.Module`).
Custom data layouts and detection metrics are v2.
