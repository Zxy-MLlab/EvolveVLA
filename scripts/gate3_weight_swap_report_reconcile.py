#!/usr/bin/env python3
"""Reconcile checkpoint keys using the official PI05 loader's canonical names."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path("/root/autodl-tmp/EvolveVLA")
JSON_PATH = ROOT / "results/gate3_weight_swap_fixed_config.json"
REPORT_PATH = ROOT / "results/gate3_weight_swap_fixed_config.md"


def canonical_official(key: str, shape: list[int]) -> dict[str, list[list[int]]]:
    if key.startswith("state_proj."):
        return {}
    mapped = key
    if mapped.startswith("action_time_mlp_in."):
        mapped = mapped.replace("action_time_mlp_in.", "time_mlp_in.", 1)
    elif mapped.startswith("action_time_mlp_out."):
        mapped = mapped.replace("action_time_mlp_out.", "time_mlp_out.", 1)
    if mapped.startswith("model."):
        canonical = mapped
    else:
        canonical = f"model.{mapped}"
    result = {canonical: [shape]}
    if key in {
        "model.paligemma_with_expert.paligemma.lm_head.weight",
        "paligemma_with_expert.paligemma.lm_head.weight",
    }:
        result[
            "model.paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        ] = [shape]
    return result


def reconstruct_raw_maps(audit: dict[str, Any]) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    official: dict[str, list[int]] = {}
    sidney: dict[str, list[int]] = {}
    for item in audit["official_only"]:
        official[item["key"]] = item["official"]
    for item in audit["sidney_only"]:
        sidney[item["key"]] = item["sidneyxie"]
    for item in audit.get("shape_mismatch", []):
        official[item["key"]] = item["official"]
        sidney[item["key"]] = item["sidneyxie"]
    return official, sidney


def canonical_audit(payload: dict[str, Any]) -> dict[str, Any]:
    raw_official, raw_sidney = reconstruct_raw_maps(payload["weight_audit"])
    official: dict[str, list[int]] = {}
    for key, shape in raw_official.items():
        for mapped, shapes in canonical_official(key, shape).items():
            official[mapped] = shapes[0]
    sidney = raw_sidney
    common = set(official) & set(sidney)
    same = sorted(key for key in common if official[key] == sidney[key])
    mismatch = sorted(
        {
            "key": key,
            "official": official[key],
            "sidneyxie": sidney[key],
        }
        for key in common
        if official[key] != sidney[key]
    )
    return {
        "raw_official_keys": len(raw_official),
        "raw_sidney_keys": len(raw_sidney),
        "canonical_official_keys": len(official),
        "canonical_sidney_keys": len(sidney),
        "canonical_common_keys": len(common),
        "canonical_same_shape_keys": len(same),
        "canonical_same_shape_sample": same[:10],
        "canonical_shape_mismatch": mismatch,
        "canonical_official_only": [
            {"key": key, "official": official[key]}
            for key in sorted(set(official) - set(sidney))
        ],
        "canonical_sidney_only": [
            {"key": key, "sidneyxie": sidney[key]}
            for key in sorted(set(sidney) - set(official))
        ],
        "handling": (
            "Official raw keys were compared after the loader's canonical "
            "transform: optional action_time_mlp_in/out rename, then model. "
            "prefix. SidneyXie keys were already canonical. Weight injection "
            "used exact canonical policy keys only; no tensor-shape map was used."
        ),
    }


def write_report(payload: dict[str, Any], audit: dict[str, Any]) -> None:
    oracle = payload["oracle"]
    teacher = payload["teacher_forcing"]
    inject = payload["model_inject_audit"]
    lines = [
        "# Gate3 Weight Swap Under Fixed Official Config",
        "",
        "- Experiment: `gate3_weight_swap_fixed_config`",
        "- Parent: `gate3_official_protocol_5x400_armfix`",
        f"- Harness SHA256: `{payload['harness_sha256']}`",
        f"- Dataset sample: episode `{payload['episode']}`, position "
        f"`{payload['position']}`, video index `{payload['video_index']}`",
        "- No simulator rollout was performed.",
        "",
        "## Fixed Configuration",
        "",
        "- `harness.load_policy` loaded the official config and all processors "
        "from `checkpoints/pi05_robotwin_lerobot`.",
        "- Verified normalization: `VISUAL=IDENTITY, STATE=QUANTILES, "
        "ACTION=QUANTILES`.",
        "- Only SidneyXie `model.safetensors` tensors were injected into the "
        "already-constructed official-config policy.",
        "",
        "## Canonical Key And Shape Audit",
        "",
        f"- Raw official keys: `{audit['raw_official_keys']}`",
        f"- Raw SidneyXie keys: `{audit['raw_sidney_keys']}`",
        f"- Canonical official keys after loader remap: "
        f"`{audit['canonical_official_keys']}`",
        f"- Canonical SidneyXie keys: `{audit['canonical_sidney_keys']}`",
        f"- Canonical common keys: `{audit['canonical_common_keys']}`",
        f"- Same-shape canonical keys: `{audit['canonical_same_shape_keys']}`",
        f"- Canonical shape mismatches: `{len(audit['canonical_shape_mismatch'])}`",
        f"- Canonical official-only keys: `{len(audit['canonical_official_only'])}`",
        f"- Canonical SidneyXie-only keys: `{len(audit['canonical_sidney_only'])}`",
        f"- Same-shape sample: `{audit['canonical_same_shape_sample']}`",
        "",
        f"- Handling: {audit['handling']}",
        f"- Exact policy-state tensors copied: `{inject['loaded_keys']}`",
        f"- Policy-state shape skips: `{len(inject['model_shape_mismatch'])}`",
        "",
        "### Canonical Shape Mismatches",
        "",
    ]
    if audit["canonical_shape_mismatch"]:
        for item in audit["canonical_shape_mismatch"]:
            lines.append(
                f"- `{item['key']}`: official `{item['official']}` vs "
                f"SidneyXie `{item['sidneyxie']}`; **SKIP**"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "### Canonical Official-Only Keys", ""])
    if audit["canonical_official_only"]:
        for item in audit["canonical_official_only"]:
            lines.append(f"- `{item['key']}` shape `{item['official']}`; **SKIP**")
    else:
        lines.append("- None.")
    lines.extend(["", "### Canonical SidneyXie-Only Keys", ""])
    if audit["canonical_sidney_only"]:
        for item in audit["canonical_sidney_only"]:
            lines.append(
                f"- `{item['key']}` shape `{item['sidneyxie']}`; **SKIP**"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Oracle",
            "",
            f"- Official-config + SidneyXie-weights arm L2: "
            f"`{oracle['final_arm_l2']:.8f}`",
            "- Comparison: official weights `0.0598`; SidneyXie own-config "
            "reference `3.242`; recorded action `3.294`.",
            "- Interpretation: `1.10712624` is far above the official mean-like "
            "output but below the `~3.2` SidneyXie reference; this is a "
            "weights-side discriminator result with partial action-scale "
            "recovery, not full behavioral recovery.",
            f"- Raw arm L2: `{oracle['raw_arm_l2']:.8f}`",
            f"- Final action: `{oracle['final_action']}`",
            "",
            "## Teacher Forcing",
            "",
            "| t | cosine(v_t, u_t) | MSE(v_t, u_t) | L2(v_t-u_t) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for time_value in ("0.5", "0.1"):
        metric = teacher[time_value]
        lines.append(
            f"| {time_value} | `{metric['cosine']:.8f}` | "
            f"`{metric['mse']:.8f}` | `{metric['l2']:.8f}` |"
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- **{payload['verdict']}**",
            f"- Named next step: {payload['next_step']}",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    audit = canonical_audit(payload)
    payload["canonical_key_audit"] = audit
    payload["report_reconciled"] = True
    payload["report_reconcile_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(payload, audit)
    print(f"CANONICAL_SAME_SHAPE_KEYS={audit['canonical_same_shape_keys']}")
    print(f"CANONICAL_SHAPE_MISMATCH_KEYS={len(audit['canonical_shape_mismatch'])}")
    print(f"CANONICAL_OFFICIAL_ONLY_KEYS={len(audit['canonical_official_only'])}")
    print(f"CANONICAL_SIDNEY_ONLY_KEYS={len(audit['canonical_sidney_only'])}")
    print(f"REPORT={REPORT_PATH}")
    print(
        f"REPORT_SHA256={hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
