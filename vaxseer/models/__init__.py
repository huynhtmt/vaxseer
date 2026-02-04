# vaxseer/models/__init__.py
"""isort:skip_file"""
import warnings
import importlib
import os

MODEL_REGISTRY = {}


def register_model(name: str):
    """
    Back-compat public decorator:
        @register_model("gpt2_time_new")
        class GPT2TimeNew(...): ...
    """
    return register_model_cls(name)


def register_model_cls(name: str):
    """
    Internal decorator used by register_model.
    Duplicate registrations warn and return the already-registered class.
    """
    def wrapper(cls):
        if name in MODEL_REGISTRY:
            warnings.warn(
                f"Model '{name}' already registered; skipping duplicate.",
                RuntimeWarning
            )
            return MODEL_REGISTRY[name]
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def build_model(name: str):
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise ValueError(f"Cannot find model named {name}")


def import_models(models_dir: str, namespace: str):
    """
    Dynamically import every .py (and package) in models_dir under 'namespace'.
    """
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + module)


# automatically import any Python files in this directory as 'vaxseer.models.<mod>'
models_dir = os.path.dirname(__file__)
import_models(models_dir, __name__)
