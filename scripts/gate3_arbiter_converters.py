import numpy as np
from safetensors import safe_open


SOURCE_PATH = "/root/autodl-tmp/EvolveVLA/checkpoints/convert_stageA/orbax_params.npz"
OURS_PATH = "/root/autodl-tmp/EvolveVLA/checkpoints/pi05_robotwin_lerobot/model.safetensors"
CANONICAL_PATH = "/root/autodl-tmp/EvolveVLA/checkpoints/pi05_robotwin_CANONICAL/model.safetensors"
TOLERANCE = 1e-6

SOURCE_BASE = (
    "dict['params']/dict['PaliGemma']/dict['img']/dict['Transformer']"
    "/dict['encoderblock']"
)


def source_leaf(*parts):
    return "/".join([SOURCE_BASE] + [f"dict['{part}']" for part in parts] + ["dict['value']"])


def source_mapping(key):
    prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers."
    if not key.startswith(prefix):
        raise KeyError(key)
    remainder = key[len(prefix):]
    layer_text, tensor_name = remainder.split(".", 1)
    layer = int(layer_text)
    if tensor_name == "layer_norm2.bias":
        return layer, source_leaf("LayerNorm_1", "bias")
    if tensor_name == "layer_norm2.weight":
        return layer, source_leaf("LayerNorm_1", "scale")
    if tensor_name == "mlp.fc2.bias":
        return layer, source_leaf("MlpBlock_0", "Dense_1", "bias")
    raise KeyError(key)


def short_values(array):
    return np.array2string(
        np.asarray(array).reshape(-1)[:5],
        precision=7,
        separator=",",
        suppress_small=False,
    )


def main():
    source = np.load(SOURCE_PATH)
    keys = [
        *[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{layer}.layer_norm2.{kind}"
            for layer in (21, 22, 23, 24)
            for kind in ("bias", "weight")
        ],
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.26.mlp.fc2.bias",
    ]

    print(f"SOURCE_LEAF_NAMES ({len(source.files)}):")
    for leaf_name in source.files:
        print(f"  {leaf_name}")

    print("\nKEY_MAPPING:")
    mappings = {}
    for key in keys:
        layer, leaf_name = source_mapping(key)
        mappings[key] = (layer, leaf_name)
        print(f"  {key} -> {leaf_name}[{layer}]")

    with safe_open(OURS_PATH, framework="pt", device="cpu") as ours_file:
        with safe_open(CANONICAL_PATH, framework="pt", device="cpu") as canonical_file:
            print("\nARBITRATION:")
            for key in keys:
                layer, leaf_name = mappings[key]
                source_value = np.asarray(source[leaf_name][layer])
                ours_value = ours_file.get_tensor(key).detach().cpu().numpy()
                canonical_value = canonical_file.get_tensor(key).detach().cpu().numpy()

                ours_diff = float(np.max(np.abs(ours_value - source_value)))
                canonical_diff = float(np.max(np.abs(canonical_value - source_value)))
                ours_match = ours_diff <= TOLERANCE
                canonical_match = canonical_diff <= TOLERANCE
                if ours_match and canonical_match:
                    verdict = "BOTH"
                elif ours_match:
                    verdict = "OURS"
                elif canonical_match:
                    verdict = "CANONICAL"
                else:
                    verdict = "NEITHER"
                bitwise = (
                    f"OURS={'yes' if np.array_equal(ours_value, source_value) else 'no'},"
                    f"CANONICAL={'yes' if np.array_equal(canonical_value, source_value) else 'no'}"
                )
                print(
                    f"{key} | shape={source_value.shape} | "
                    f"SOURCE={short_values(source_value)} | "
                    f"OURS={short_values(ours_value)} | "
                    f"CANONICAL={short_values(canonical_value)} | "
                    f"match={verdict} | {bitwise} | "
                    f"max|OURS-SRC|={ours_diff:.9g} | "
                    f"max|CAN-SRC|={canonical_diff:.9g}"
                )


if __name__ == "__main__":
    main()
