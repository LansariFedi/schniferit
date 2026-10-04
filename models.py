import importlib
import torch
import torchvision.models as tvmodels


def find_tv_weights(arch):
    key = arch.lower().replace("_", "")
    for name in dir(tvmodels):
        if name.endswith("_Weights") and name[:-8].lower().replace("_", "") == key:
            return getattr(tvmodels, name)
    return None


def swap_head(model, num_classes):
    if isinstance(getattr(model, "fc", None), torch.nn.Linear):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        return model
    c = getattr(model, "classifier", None)
    if isinstance(c, torch.nn.Sequential):
        for i in reversed(range(len(c))):
            if isinstance(c[i], torch.nn.Linear):
                c[i] = torch.nn.Linear(c[i].in_features, num_classes)
                return model
    elif isinstance(c, torch.nn.Linear):
        model.classifier = torch.nn.Linear(c.in_features, num_classes)
        return model
    h = getattr(getattr(model, "heads", None), "head", None)
    if isinstance(h, torch.nn.Linear):
        model.heads.head = torch.nn.Linear(h.in_features, num_classes)
        return model
    raise ValueError(
        f"no known head pattern on {type(model).__name__}; use --arch pkg:factory"
    )


def head_parameters(model):
    if isinstance(getattr(model, "fc", None), torch.nn.Linear):
        yield from model.fc.parameters()
        return
    c = getattr(model, "classifier", None)
    if isinstance(c, torch.nn.Sequential):
        for m in reversed(list(c)):
            if isinstance(m, torch.nn.Linear):
                yield from m.parameters()
                return
    elif isinstance(c, torch.nn.Linear):
        yield from c.parameters()
        return
    h = getattr(getattr(model, "heads", None), "head", None)
    if isinstance(h, torch.nn.Linear):
        yield from h.parameters()
        return
    raise ValueError(f"no known head pattern on {type(model).__name__}")


def build_model(arch, num_classes, pretrained=False):
    if ":" in arch:
        mod_name, fn_name = arch.split(":")
        return getattr(importlib.import_module(mod_name), fn_name)(num_classes)
    if not hasattr(tvmodels, arch):
        raise ValueError(f"unknown arch {arch!r}")
    factory = getattr(tvmodels, arch)
    if pretrained:
        w = find_tv_weights(arch)
        model = factory(weights=(w.DEFAULT if w else None))
    else:
        model = factory(weights=None)
    return swap_head(model, num_classes)
