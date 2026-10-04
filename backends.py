import importlib
import torch


def probe_backends():
    rows = [("torch.compile", hasattr(torch, "compile"), torch.__version__)]
    rows.append(
        (
            "cuda",
            torch.cuda.is_available(),
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU",
        )
    )
    try:
        rows.append(("quantized_cpu", True, torch.backends.quantized.engine))
    except Exception as e:
        rows.append(("quantized_cpu", False, str(e)))
    for mod in ["onnx", "onnxruntime", "tensorrt", "openvino", "torchao"]:
        try:
            m = importlib.import_module(mod)
            extra = (
                ",".join(m.get_available_providers()) if mod == "onnxruntime" else ""
            )
            rows.append(
                (mod, True, (getattr(m, "__version__", "") + " " + extra).strip())
            )
        except Exception:
            rows.append((mod, False, "not installed"))
    return rows
