from safetensors.torch import load_file


OURS_PATH = (
    "/root/autodl-tmp/EvolveVLA/checkpoints/"
    "pi05_robotwin_lerobot/model.safetensors"
)
CANONICAL_PATH = (
    "/root/autodl-tmp/EvolveVLA/checkpoints/"
    "pi05_robotwin_CANONICAL/model.safetensors"
)


def main():
    ours = load_file(OURS_PATH, device="cpu")
    canonical = load_file(CANONICAL_PATH, device="cpu")

    ours_keys = set(ours)
    canonical_keys = set(canonical)
    common = ours_keys & canonical_keys
    ours_only = sorted(ours_keys - canonical_keys)
    canonical_only = sorted(canonical_keys - ours_keys)

    print(f"OURS key count: {len(ours_keys)}")
    print(f"CANONICAL key count: {len(canonical_keys)}")
    print(f"ours_only: {ours_only}")
    print(f"canonical_only: {canonical_only}")

    diffs = []
    shape_mismatches = 0
    for key in sorted(common):
        ours_tensor = ours[key]
        canonical_tensor = canonical[key]
        if ours_tensor.shape != canonical_tensor.shape:
            shape_mismatches += 1
            print(
                "SHAPE_MISMATCH "
                f"key={key} ours_shape={tuple(ours_tensor.shape)} "
                f"canonical_shape={tuple(canonical_tensor.shape)}"
            )
            continue

        max_abs_diff = (
            ours_tensor.float() - canonical_tensor.float()
        ).abs().max().item()
        diffs.append((max_abs_diff, key))

    diffs.sort(key=lambda item: (-item[0], item[1]))
    global_max = max((max_abs_diff for max_abs_diff, _ in diffs), default=0.0)
    n_gt_threshold = sum(
        max_abs_diff > 1e-6 for max_abs_diff, _ in diffs
    )

    print(f"n_common: {len(common)}")
    print(f"n_compared: {len(diffs)}")
    print(f"n_shape_mismatches: {shape_mismatches}")
    print(f"global max|delta|: {global_max:.10g}")
    print(f"n_common with max|delta| > 1e-6: {n_gt_threshold}")
    print("top-5 worst keys:")
    for max_abs_diff, key in diffs[:5]:
        print(f"  {key}: {max_abs_diff:.10g}")

    ours_embed_head = sorted(
        key
        for key in ours_keys
        if "embed_tokens.weight" in key or "lm_head.weight" in key
    )
    canonical_embed_head = sorted(
        key
        for key in canonical_keys
        if "embed_tokens.weight" in key or "lm_head.weight" in key
    )
    print(f"ours embed/head keys: {ours_embed_head}")
    print(f"canonical embed/head keys: {canonical_embed_head}")

    print(
        "WEIGHT_DIFF "
        f"n_common={len(common)} "
        f"global_max_absdiff={global_max:.10g} "
        f"n_gt_1e-6={n_gt_threshold} "
        f"ours_only={ours_only} "
        f"canonical_only={canonical_only}"
    )


if __name__ == "__main__":
    main()
