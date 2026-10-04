import copy
import os
import torch


class AmpWrapper(torch.nn.Module):
    def __init__(self, mod, device):
        super().__init__()
        self.m = mod
        self.device = device

    def forward(self, xb):
        with torch.amp.autocast(self.device):
            return self.m(xb)


def stage_1_compile_fp16(ctx, last_model):
    if not hasattr(torch, "compile"):
        return None, "torch.compile missing"
    if not torch.cuda.is_available():
        return None, "needs CUDA"
    try:
        m = torch.compile(copy.deepcopy(last_model).eval().to("cuda"))
        xb = ctx["sample"][:1].to("cuda")
        with torch.amp.autocast("cuda"):
            m(xb)
        torch.cuda.synchronize()
    except Exception as e:
        return None, f"compile failed: {str(e)[:100]}"
    return {
        "model": AmpWrapper(m, "cuda"),
        "target": "gpu",
        "file": None,
        "size_mb": None,
    }, "recipe, no file"


def stage_2_static_int8(ctx, _last_model):
    try:
        from torch.ao.quantization import get_default_qconfig
        from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx
    except Exception as e:
        return None, f"quant API missing: {str(e)[:80]}"
    try:
        for eng in ("fbgemm", "qnnpack"):
            try:
                torch.backends.quantized.engine = eng
                break
            except Exception:
                continue
        m = copy.deepcopy(ctx["fp32"]).cpu().eval()
        n_fp32 = sum(p.numel() for p in m.parameters())
        example = (ctx["sample"][:1],)

        mp = prepare_fx(
            m, {"": get_default_qconfig(torch.backends.quantized.engine)}, example
        )
        task = ctx["task"]
        calib = task.build_loader(
            ctx["data"], "train", ctx["tf_val"], batch_size=32, workers=2
        )
        seen = 0
        with torch.no_grad():
            for xb, _ in calib:
                mp(xb)
                seen += xb.size(0)
                if seen >= 300:
                    break
        qm = convert_fx(mp)
        path = "artifacts/stage2_int8.pt"
        os.makedirs("artifacts", exist_ok=True)
        torch.save(qm, path)
    except Exception as e:
        return None, f"quant failed: {str(e)[:100]}"
    return {
        "model": qm,
        "target": "cpu",
        "file": path,
        "params": n_fp32,
        "size_mb": os.path.getsize(path) / 1e6,
    }, f"calibrated on {seen} train images"


def stage_3_structured_prune(ctx, _last_model):
    try:
        import torch_pruning as tp
    except Exception as e:
        return None, f"torch-pruning missing: {str(e)[:80]}"
    try:
        import torch.nn as nn

        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = copy.deepcopy(ctx["fp32"]).to(device).train()
        n_before = sum(p.numel() for p in m.parameters())
        example = ctx["sample"][:1].to(device)
        ignored = [
            mod
            for mod in m.modules()
            if isinstance(mod, nn.Linear) and mod.out_features == ctx["num_classes"]
        ]
        pruner = tp.pruner.MagnitudePruner(
            m,
            example,
            importance=tp.importance.MagnitudeImportance(p=2),
            ch_sparsity=ctx["prune_ratio"],
            ignored_layers=ignored,
        )
        pruner.step()
        n_after = sum(p.numel() for p in m.parameters())
        task = ctx["task"]
        heal_loader = task.build_loader(
            ctx["data"], "train", ctx["tf_train"], batch_size=32
        )
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()
        for _ in range(ctx["heal_epochs"]):
            for xb, yb in heal_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss_fn(m(xb), yb).backward()
                opt.step()
        m.eval().cpu()
        path = "artifacts/stage3_pruned.pt"
        os.makedirs("artifacts", exist_ok=True)
        torch.save(m, path)
    except Exception as e:
        return None, f"prune failed: {str(e)[:100]}"
    return (
        {
            "model": m,
            "target": "cpu",
            "file": path,
            "params": n_after,
            "size_mb": os.path.getsize(path) / 1e6,
            "updates_fp32": True,
        },
        f"channels -{100 * (1 - n_after / n_before):.0f}%, healed {ctx['heal_epochs']} epochs",
    )


def stage_4_distill(ctx, _last_model):
    import torch.nn as nn
    import torch.nn.functional as F
    from .models import build_model

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        teacher = build_model(ctx["arch"], ctx["num_classes"])
        teacher.load_state_dict(
            torch.load(ctx["weights"], map_location="cpu", weights_only=True)
        )
        teacher = teacher.to(device).eval()
        student = (
            build_model(ctx["student"], ctx["num_classes"], pretrained=True)
            .to(device)
            .train()
        )
        task = ctx["task"]
        loader = task.build_loader(ctx["data"], "train", ctx["tf_train"], batch_size=32)
        opt = torch.optim.AdamW(student.parameters(), lr=3e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=3e-4, epochs=ctx["distill_epochs"], steps_per_epoch=len(loader)
        )
        T, alpha = 4.0, 0.7
        for _ in range(ctx["distill_epochs"]):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                with torch.no_grad():
                    soft = teacher(xb)
                out = student(xb)
                loss = alpha * F.kl_div(
                    F.log_softmax(out / T, 1),
                    F.softmax(soft / T, 1),
                    reduction="batchmean",
                ) * T * T + (1 - alpha) * F.cross_entropy(out, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        student.eval().cpu()
        path = "artifacts/stage4_student.pt"
        os.makedirs("artifacts", exist_ok=True)
        torch.save(student.state_dict(), path)
    except Exception as e:
        return None, f"distill failed: {str(e)[:100]}"
    return {
        "model": student,
        "target": "cpu",
        "file": path,
        "params": sum(p.numel() for p in student.parameters()),
        "size_mb": os.path.getsize(path) / 1e6,
        "updates_fp32": True,
        "student_arch": ctx["student"],
    }, f"student {ctx['student']}, teacher {ctx['arch']} reference"


STAGES = [
    ("1_compile_fp16", stage_1_compile_fp16, "gpu", {"classify"}),
    ("2_static_int8", stage_2_static_int8, "cpu", {"classify"}),
    ("3_structured_prune", stage_3_structured_prune, "cpu", {"classify"}),
    ("4_distill", stage_4_distill, "cpu", {"classify"}),
]
