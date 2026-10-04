import argparse
import os
import sys
import torch
from . import stages as stage_mod
from .backends import probe_backends
from .export import export_verify
from .measure import measure
from .models import build_model, find_tv_weights
from .tasks import classify

TASKS = {"classify": classify}


def cmd_list(_a):
    print(f"{'backend':<16} {'ok':<4} note")
    for name, ok, note in probe_backends():
        print(f"{name:<16} {'yes' if ok else 'no':<4} {note}")


def append_row(stage, arch, task, m, verdict):
    new = not os.path.exists("results.md")
    with open("results.md", "a") as f:
        if new:
            f.write(
                f"# schniferit ledger\n\n"
                "| stage | arch | task | acc | size_MB | params | cpu_b1_ms | gpu_b1_ms | verdict |\n"
                "|---|---|---|---|---|---|---|---|---|\n"
            )
        f.write(
            f"| {stage} | {arch} | {task} | {m['metric']:.4f} | {m['size_mb']:.1f} | "
            f"{m['params']} | {m['cpu_b1']:.2f} | {m['gpu_b1']:.2f} | {verdict} |\n"
        )


def load_fp32(a):
    task = TASKS[a.task]
    model = build_model(a.arch, a.num_classes)
    model.load_state_dict(torch.load(a.weights, map_location="cpu", weights_only=True))
    if ":" not in a.arch and (w := find_tv_weights(a.arch)):
        tf_val = w.DEFAULT.transforms()
    else:
        tf_val = task.default_val_transform(a.img_size)
    val_loader = task.build_loader(a.data, "val", tf_val)
    return task, model, tf_val, val_loader


def cmd_baseline(a):
    task, model, _tf, val_loader = load_fp32(a)
    sample = next(iter(val_loader))[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = measure(
        model,
        sample,
        val_loader,
        task.accuracy,
        weights_path=a.weights,
        devices=(device,),
    )
    append_row("baseline", a.arch, a.task, m, "reference")
    print(
        f"baseline: acc {m['metric']:.4f} | {m['size_mb']:.1f}MB | "
        f"cpu {m['cpu_b1']:.2f}ms | gpu {m['gpu_b1']:.2f}ms → results.md"
    )


def cmd_optimize(a):
    task, fp32, tf_val, val_loader = load_fp32(a)
    sample = next(iter(val_loader))[0]
    if ":" not in a.arch and (w := find_tv_weights(a.arch)):
        preset = w.DEFAULT.transforms()
        mean, std = preset.mean, preset.std
        size = (
            preset.crop_size[0]
            if isinstance(preset.crop_size, (list, tuple))
            else preset.crop_size
        )
    else:
        mean, std, size = task.IMAGENET_MEAN, task.IMAGENET_STD, a.img_size
    tf_train = task.default_train_transform(size, mean, std)
    ref = measure(
        fp32,
        sample,
        val_loader,
        task.accuracy,
        weights_path=a.weights,
        devices=("cpu", "cuda"),
    )
    print(
        f"reference: acc {ref['metric']:.4f} | {ref['size_mb']:.1f}MB | "
        f"cpu {ref['cpu_b1']:.2f}ms | gpu {ref['gpu_b1']:.2f}ms | floor {ref['metric'] - a.budget / 100:.4f}"
    )
    ctx = {
        "sample": sample,
        "tf_val": tf_val,
        "tf_train": tf_train,
        "task": task,
        "data": a.data,
        "fp32": fp32,
        "num_classes": a.num_classes,
        "arch": a.arch,
        "weights": a.weights,
        "student": a.student,
        "prune_ratio": a.prune_ratio,
        "heal_epochs": a.heal_epochs,
        "distill_epochs": a.distill_epochs,
    }
    want = [s.strip() for s in a.stages.split(",")]
    last_model, accepted = fp32, {"ref": ref}
    for name, fn, target, supports in stage_mod.STAGES:
        if want != ["all"] and name.split("_")[0] not in want and name not in want:
            continue
        if a.task not in supports:
            append_row(name, a.arch, a.task, ref, f"SKIP (needs {sorted(supports)})")
            print(f"[{name}] SKIP (needs {sorted(supports)})")
            continue
        art, note = fn(ctx, last_model)
        if art is None:
            append_row(name, a.arch, a.task, ref, f"REVERT ({note})")
            print(f"[{name}] REVERT ({note})")
            continue
        m = measure(
            art["model"],
            sample,
            val_loader,
            task.accuracy,
            weights_path=art["file"] or a.weights,
            size_mb=art["size_mb"] if art["size_mb"] is not None else ref["size_mb"],
            devices=({"cpu": "cpu", "gpu": "cuda"}[target],),
        )
        drop_pt = (ref["metric"] - m["metric"]) * 100
        if art.get("params"):
            m["params"] = art["params"]
        lat_key = f"{target}_b1"
        if drop_pt > a.budget:
            verdict = f"REVERT (over budget: -{drop_pt:.1f}pt > {a.budget})"
        elif m[lat_key] < ref[lat_key] or m["size_mb"] < ref["size_mb"]:
            verdict = (
                f"ACCEPT (Δ-{drop_pt:.1f}pt, {target} {ref[lat_key]:.2f}→{m[lat_key]:.2f}ms, "
                f"{ref['size_mb']:.1f}→{m['size_mb']:.1f}MB)"
            )
            last_model, accepted[name] = art["model"], m
            if art.get("updates_fp32"):
                ctx["fp32"] = art["model"]
            if art.get("file"):
                _opath, onote = export_verify(art["model"], sample, name)
                verdict += f" + {onote}"
        else:
            verdict = "REVERT (no gain)"
        append_row(name, a.arch, a.task, m, verdict)
        print(f"[{name}] {verdict} | {note}")
    for target in ("cpu", "gpu"):
        key = f"{target}_b1"
        avail = {n: m for n, m in accepted.items() if m[key] == m[key]}
        win = min(avail, key=lambda n: avail[n][key])
        print(
            f"WINNER[{target}]: {win} — {avail[win][key]:.2f}ms, acc {avail[win]['metric']:.4f}"
        )


def main():
    p = argparse.ArgumentParser(
        description="schniferit: squeeze any model, gated by budget."
    )
    sub = p.add_subparsers(required=True)
    sub.add_parser(
        "list", help="probe available backends on this machine"
    ).set_defaults(fn=cmd_list)
    b = sub.add_parser("baseline", help="measure a weights file → results.md row 0")
    b.add_argument("--weights", default="artifacts/best.pt")
    b.add_argument("--arch", default="resnet18")
    b.add_argument("--task", default="classify", choices=list(TASKS))
    b.add_argument("--data", default="PetImages")
    b.add_argument("--num-classes", type=int, default=2)
    b.add_argument("--img-size", type=int, default=224)
    b.set_defaults(fn=cmd_baseline)
    o = sub.add_parser("optimize", help="run the gated stage ladder")
    o.add_argument("--weights", default="artifacts/best.pt")
    o.add_argument("--arch", default="resnet18")
    o.add_argument("--task", default="classify", choices=list(TASKS))
    o.add_argument("--data", default="PetImages")
    o.add_argument("--num-classes", type=int, default=2)
    o.add_argument("--img-size", type=int, default=224)
    o.add_argument("--budget", type=float, default=3.0, help="max acc drop, points")
    o.add_argument("--stages", default="all", help="'all' or comma list e.g. '1,2'")
    o.add_argument(
        "--prune-ratio", type=float, default=0.3, help="channel sparsity for stage 3"
    )
    o.add_argument(
        "--heal-epochs", type=int, default=3, help="finetune epochs after pruning"
    )
    o.add_argument(
        "--distill-epochs", type=int, default=12, help="student training epochs"
    )
    o.add_argument(
        "--student", default="mobilenet_v3_small", help="student arch for stage 4"
    )
    o.set_defaults(fn=cmd_optimize)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
