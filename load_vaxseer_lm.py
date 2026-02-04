
# load_vaxseer_lm.py  (PATCHED FRAGMENTS)

import os
import sys
import inspect
from typing import Tuple, Dict, Any, Iterable, Optional

import torch



def _maybe_add_to_syspath(vaxseer_root: Optional[str] = None):
    """
    Ensure the VaxSeer package root is importable.
    If Shiny/global.R already did 'sys.path.insert(0, VAXSEER_ROOT)', this is a no-op.
    """
    if vaxseer_root and os.path.isdir(vaxseer_root):
        if vaxseer_root not in sys.path:
            sys.path.insert(0, vaxseer_root)


def _extract_hparams(ckpt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Lightning 1.x most often stores hyper-parameters under 'hyper_parameters' or 'hparams'.
    Return {} if not present.
    """
    if isinstance(ckpt, dict):
        if "hyper_parameters" in ckpt and isinstance(ckpt["hyper_parameters"], dict):
            return dict(ckpt["hyper_parameters"])
        if "hparams" in ckpt and isinstance(ckpt["hparams"], dict):
            return dict(ckpt["hparams"])
    return {}


def _filter_kwargs_for_ctor(ctor, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep only kwargs that appear in the model __init__ signature.
    This prevents unexpected-arg errors if the checkpoint contains extra config fields.
    """
    sig = inspect.signature(ctor)
    valid = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in valid}


def _clean_state_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Many training setups save keys as 'model.xxx', 'module.xxx', etc.
    Strip the most common prefixes so that keys match the target module names.
    """
    def strip_prefix(k: str, p: str) -> str:
        return k[len(p):] if k.startswith(p) else k

    cleaned = {}
    for k, v in sd.items():
        k2 = k
        for prefix in ("model.", "module.", "net.", "network."):
            k2 = strip_prefix(k2, prefix)
        cleaned[k2] = v
    return cleaned


def _load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Use torch.load to get the full checkpoint dict.
    (We DO NOT pass weights_only=True because we need hyper_parameters as well.)
    """
    return torch.load(path, map_location=map_location)


def load_gpt2_time_from_checkpoint(
    ckpt_path: str,
    vaxseer_root: Optional[str] = None,
    map_location: str = "cpu",
    strict: bool = False
):
    """
    Rebuild authors' GPT2TimeModel and load weights — without Lightning's classmethod.

    Parameters
    ----------
    ckpt_path : str
        Path to the .ckpt file from runs/flu_lm/...
    vaxseer_root : str or None
        Optional absolute path to VaxSeer repo root.
        If provided, will be added to sys.path for imports.
    map_location : str
        'cpu' (default) or e.g. 'cuda:0' if you have a GPU environment.
    strict : bool
        Whether to enforce exact key matching in load_state_dict.

    Returns
    -------
    model : torch.nn.Module
        An instance of vaxseer.models.gpt2_time.GPT2TimeModel in eval() mode.
    info  : dict
        Diagnostics: {'missing_keys': [...], 'unexpected_keys': [...], 'used_hparams': {...}}
    """
    _maybe_add_to_syspath(vaxseer_root)

    # Import here after sys.path adjustments
    from vaxseer.models.gpt2_time import GPT2TimeModel

    # 1) Read checkpoint fully (need hparams)
    ckpt = _load_checkpoint(ckpt_path, map_location=map_location)

    # 2) Pull hyper-parameters
    hparams = _extract_hparams(ckpt)

    # 3) Filter hparams to match GPT2TimeModel.__init__
    ctor_kwargs = _filter_kwargs_for_ctor(GPT2TimeModel.__init__, hparams)

    # 4) Construct the model
    model = GPT2TimeModel(**ctor_kwargs)

    # 5) Prepare weights
    if "state_dict" not in ckpt or not isinstance(ckpt["state_dict"], dict):
        raise RuntimeError(
            f"Checkpoint at {ckpt_path} does not contain a valid 'state_dict' (found keys: {list(ckpt.keys())[:10]}...)"
        )
    state_dict = _clean_state_keys(ckpt["state_dict"])

    # 6) Load weights
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    model.eval()

    info = {
        "missing_keys": list(missing) if isinstance(missing, Iterable) else [],
        "unexpected_keys": list(unexpected) if isinstance(unexpected, Iterable) else [],
        "used_hparams": ctor_kwargs
    }
    return model, info


# ---- Convenience scoring wrapper (optional) ----------------------------------

def score_sequences_if_available(model, aa_sequences):
    """
    If the model exposes a 'score_sequences' method (some authors do), call it.
    Otherwise raise a clear error so you can implement your scoring path.
    """
    if hasattr(model, "score_sequences") and callable(getattr(model, "score_sequences")):
        with torch.no_grad():
            return model.score_sequences(aa_sequences)
    raise NotImplementedError(
        "This model doesn't expose 'score_sequences'. "
        "Use the project's Trainer.predict(...) path (Option B) or "
        "adapt this function to your model's forward/batching logic."
    )

def _enable_safe_globals():
    """
    For PyTorch >= 2.6 (weights_only=True path), allowlist classes that appear in the checkpoint
    so torch.load can safely unpickle them.
    If this fails (older torch), we silently ignore and fall back.
    """
    try:
        from torch.serialization import add_safe_globals
        # Allowlist the HF class that your ckpt references
        from transformers.models.auto.modeling_auto import AutoModelForCausalLM
        add_safe_globals([AutoModelForCausalLM])
    except Exception:
        # Older torch or missing transformers: ignore; we'll fall back to weights_only=False path.
        pass


def _load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    """
    Robust checkpoint loader:
      1) Prefer full unpickle: weights_only=False (needed to read hyper_parameters).
      2) If that fails under torch>=2.6, allowlist needed globals and retry with weights_only=True.
    """
    # First, try full load so we can access hyper_parameters
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # torch>=2.6
    except TypeError:
        # Older torch doesn't support weights_only kwarg
        return torch.load(path, map_location=map_location)
    except Exception as e:
        # If full unpickle fails (e.g., safety restriction), try safe-globals + weights_only=True
        _enable_safe_globals()
        # Retry in "safe" mode to unblock de/serialization
        return torch.load(path, map_location=map_location, weights_only=True)
