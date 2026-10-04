import os
import statistics
import time
import torch


def size_mb(path):
    return os.path.getsize(path) / 1e6


def param_count(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def latency_ms(model, sample, device, warmup=20, reps=100):
    model.eval().to(device)
    xb = sample.to(device)
    for _ in range(warmup):
        model(xb)
    if device == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        model(xb)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return statistics.median(ts)


def measure(
    model,
    sample,
    val_loader,
    metric_fn,
    weights_path=None,
    size_mb=None,
    devices=("cpu",),
):
    model.eval()
    model.to(devices[0])
    out = {
        "metric": metric_fn(model, val_loader, devices[0]),
        "params": param_count(model),
    }
    out["size_mb"] = (
        size_mb if size_mb is not None else os.path.getsize(weights_path) / 1e6
    )
    for d, k in (("cpu", "cpu_b1"), ("cuda", "gpu_b1")):
        if d in devices and (d == "cpu" or torch.cuda.is_available()):
            out[k] = latency_ms(model, sample[:1], d)
        else:
            out[k] = float("nan")
    return out
