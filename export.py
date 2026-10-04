import copy
import os
import torch


def export_verify(model, sample, name):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        return None, "onnx/ort missing"
    try:
        m = copy.deepcopy(model).cpu().eval()
        xb = sample[:1].cpu()
        path = f"artifacts/{name}.onnx"
        os.makedirs("artifacts", exist_ok=True)
        with torch.no_grad():
            torch.onnx.export(
                m,
                xb,
                path,
                opset_version=17,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
            ref = m(xb).numpy()
        got = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(
            None, {"input": xb.numpy()}
        )[0]
        diff = float(abs(ref - got).max())
        if diff > 1e-3:
            return None, f"onnx verify failed: maxdiff {diff:.2e}"
        return path, f"onnx ok (maxdiff {diff:.2e})"
    except Exception as e:
        return None, f"onnx skipped: {str(e)[:80]}"
