# vaxseer/data/data_modules/__init__.py
"""isort:skip_file"""
import warnings
import importlib
import os

DM_REGISTRY = {}


def register_dm(name: str):
    """
    Public decorator for data modules:
        @register_dm("lm_weighted")
        class ProteinLMWeightedDataModule(...): ...
    """
    return register_dm_cls(name)


def register_dm_cls(name: str):
    """
    Internal decorator backing register_dm.
    Duplicate registrations warn and return the existing class.
    """
    def wrapper(cls):
        if name in DM_REGISTRY:
            warnings.warn(
                f"Data module '{name}' already registered; skipping duplicate.",
                RuntimeWarning
            )
            return DM_REGISTRY[name]
        DM_REGISTRY[name] = cls
        return cls
    return wrapper


def build_data_module(name: str):
    if name in DM_REGISTRY:
        return DM_REGISTRY[name]
    raise ValueError(f"Cannot find data module named {name}")


def import_data_modules(dm_dir: str, namespace: str):
    """
    Dynamically import every .py (and package) in dm_dir under 'namespace'.
    """
    for file in os.listdir(dm_dir):
        path = os.path.join(dm_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + module)


# auto-import everything in this folder as 'vaxseer.data.data_modules.<mod>'
dm_dir = os.path.dirname(__file__)
import_data_modules(dm_dir, __name__)
