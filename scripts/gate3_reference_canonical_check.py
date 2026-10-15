#!/usr/bin/env python3
import importlib.util
from pathlib import Path


ROOT = Path("/root/autodl-tmp/EvolveVLA")
PINNED = ROOT / "scripts/gate3_reference_forward.py"
CANONICAL_MODEL = (
    ROOT / "checkpoints/pi05_robotwin_CANONICAL" / "model.safetensors"
)


def load_pinned_module():
    spec = importlib.util.spec_from_file_location("gate3_reference_forward", PINNED)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load pinned module from {PINNED}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_pinned_module()
    original_load_weights = module.load_weights

    def load_canonical_weights(model):
        if not CANONICAL_MODEL.is_file():
            raise FileNotFoundError(f"canonical model missing: {CANONICAL_MODEL}")
        try:
            state_dict = module.load_file(str(CANONICAL_MODEL), device="cpu")
            canonical_key = "paligemma_with_expert.paligemma.model.embed_tokens.weight"
            expected_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
            if canonical_key in state_dict and expected_key not in state_dict:
                state_dict[expected_key] = state_dict.pop(canonical_key)
            report = model.load_state_dict(state_dict, strict=False)
        except Exception as exc:
            raise RuntimeError(
                f"key_mismatch: {type(exc).__name__}: {exc}"
            ) from exc

        missing = sorted(report.missing_keys)
        unexpected = sorted(report.unexpected_keys)
        allowed_missing = {
            "paligemma_with_expert.gemma_expert.lm_head.weight",
            "paligemma_with_expert.paligemma.lm_head.weight",
        }
        if not set(missing).issubset(allowed_missing) or unexpected:
            raise RuntimeError(
                f"key_mismatch: missing={missing} unexpected={unexpected}"
            )
        return len(state_dict) - len(unexpected)

    module.load_weights = load_canonical_weights
    module.main()
    module.load_weights = original_load_weights


if __name__ == "__main__":
    main()
