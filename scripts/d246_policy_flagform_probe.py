#!/usr/bin/env python3
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["LOCAL_RANK"] = "-1"

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import PI05Config


P = "/root/autodl-tmp/EvolveVLA/checkpoints/sidneyxie_pi05_robotwin_localpre"
DATASET_ARGS = [
    "--dataset.repo_id",
    "robotwin_n3_converted",
    "--dataset.root",
    "/root/autodl-tmp/EvolveVLA/data/robotwin_n3_converted",
]
RESULT = Path("/root/autodl-tmp/EvolveVLA/results/d246_policy_flag_form.json")
LOG = Path("/root/runs/zhouxueyangapexinai/d246_policy_flagform_probe.log")


class Tee:
    def __init__(self, stream, capture):
        self.stream = stream
        self.capture = capture

    def write(self, text):
        self.stream.write(text)
        self.capture.write(text)
        return len(text)

    def flush(self):
        self.stream.flush()


def policy_type(policy):
    value = getattr(policy, "type", None)
    return value if value is not None else ("pi05" if isinstance(policy, PI05Config) else None)


def has_mean_std(value):
    return "MEAN_STD" in json.dumps(value, default=str)


def parse_form(form, argv):
    old_argv = sys.argv
    sys.argv = ["d246_policy_flagform_probe.py", *argv]
    try:
        @parser.wrap()
        def read_config(cfg: TrainPipelineConfig):
            return cfg

        config = read_config()
        config._resolve_pretrained_from_cli()
        policy = config.policy
        resolved_type = policy_type(policy) if policy is not None else None
        resolved_path = str(getattr(policy, "pretrained_path", None)) if policy else None
        resolved_norm = getattr(policy, "normalization_mapping", None) if policy else None
        return {
            "form": form,
            "ok": True,
            "policy_type": resolved_type,
            "pretrained_path": resolved_path,
            "norm": resolved_norm,
        }
    except (SystemExit, Exception) as exc:
        return {
            "form": form,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "policy_type": None,
            "pretrained_path": None,
            "norm": None,
        }
    finally:
        sys.argv = old_argv


def main():
    forms = {
        "A": ["--policy", P, *DATASET_ARGS],
        "B": ["--policy.path", P, *DATASET_ARGS],
        "C": [
            "--policy.type",
            "pi05",
            "--policy.pretrained_path",
            P,
            *DATASET_ARGS,
        ],
    }
    policy_tokens = {
        "A": ["--policy", P],
        "B": ["--policy.path", P],
        "C": ["--policy.type", "pi05", "--policy.pretrained_path", P],
    }

    capture = io.StringIO()
    results = {}
    with redirect_stdout(Tee(sys.stdout, capture)), redirect_stderr(Tee(sys.stderr, capture)):
        for form, argv in forms.items():
            result = parse_form(form, argv)
            results[form] = result
            suffix = "SUCCESS" if result["ok"] else f"EXCEPTION {result['error']}"
            print(f"FORM {form} parse {suffix}")
            if result["ok"]:
                print(
                    f"FORM {form} resolved "
                    f"policy.type={result['policy_type']} "
                    f"pretrained_path={result['pretrained_path']} "
                    f"normalization_mapping={result['norm']}"
                )

        chosen = "NONE"
        chosen_result = None
        reason = (
            "no form met parse success + type pi05 + path P + normalization mapping indicating MEAN_STD"
        )
        for form in ("A", "B", "C"):
            result = results[form]
            if (
                result["ok"]
                and result["policy_type"] == "pi05"
                and result["pretrained_path"] == P
                and has_mean_std(result["norm"])
            ):
                chosen = form
                chosen_result = result
                reason = "matched parse success + type pi05 + path P + normalization mapping indicating MEAN_STD"
                break

        print(
            "CHOSEN_ARGV_POLICY_TOKENS",
            *(policy_tokens[chosen] if chosen != "NONE" else []),
        )

        if chosen_result is None:
            resolved_type = None
            resolved_path = None
            resolved_norm = None
        else:
            resolved_type = chosen_result["policy_type"]
            resolved_path = chosen_result["pretrained_path"]
            resolved_norm = chosen_result["norm"]

        print(
            "RESOLVED",
            json.dumps(
                {
                    "chosen_form": chosen,
                    "policy.type": resolved_type,
                    "pretrained_path": resolved_path,
                    "normalization_mapping": resolved_norm,
                    "reason": reason,
                },
                default=str,
                sort_keys=True,
            ),
        )
        print("KNOWN_BLOCKER converted root lacks q01/q99; explicit MEAN_STD required for form C if used")

    payload = {
        "chosen_form": chosen,
        "argv_policy_tokens": policy_tokens[chosen] if chosen != "NONE" else [],
        "parse_ok": {form: results[form]["ok"] for form in ("A", "B", "C")},
        "resolved_policy_type": resolved_type,
        "resolved_pretrained_path": resolved_path,
        "resolved_norm": resolved_norm,
    }
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    LOG.write_text(capture.getvalue(), encoding="utf-8")


if __name__ == "__main__":
    main()
